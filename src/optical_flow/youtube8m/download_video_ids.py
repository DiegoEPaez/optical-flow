import os.path as osp
import json
import string
import random
import requests
import logging
import asyncio
import aiohttp
import shutil

from collections import deque

from optical_flow.youtube8m.utils import keep_after, random_name
from optical_flow.settings import *

log = logging.getLogger(__name__)


def read_mids(categs_file=YOUTUBE8M_CATEGORIES_FILE):
    """
    Reads mids (as specified in web page https://research.google.com/youtube8m/explore.html)
    and corresponding categories
    :param categs_file: file with categories
    :return: mids and categories as lists
    """
    mids, categories = [], []

    with open(categs_file, "r", encoding="utf-8") as file:
        while line := file.readline().rstrip():
            line_splitted = line.split()
            mids.append(line_splitted[2][1:-1])
            categories.append(' '.join(line_splitted[3:-1]))

    return mids, categories


def delete_category(category_to_delete, ids_file=YOUTUBE8M_IDS_FILE):
    """
    Given a file with the format: Category, youtube video url per line, removes all lines with
    given category in given file.

    :param category_to_delete: the category that specifies which lines will be deleted
    :param ids_file: name of the file from which lines will be removed
    :return: None
    """
    with open(ids_file, "r", encoding="utf-8") as file_read:
        temp_file = random_name(15)
        with open(temp_file, "w", encoding="utf-8") as file_write:
            while line := file_read.readline():
                if not line.startswith(category_to_delete):
                    file_write.write(line)

    shutil.move(temp_file, ids_file)


def is_finished(last_category, ids_file=YOUTUBE8M_IDS_FILE):
    """
    Determines if all youtube ids have been downloaded by checking if the given file with youtube ids
    has as last category the provided parameter_last_category
    :param last_category: The category which should be the last in the file
    :param ids_file: The file with ids
    :return:
    """
    if osp.exists(ids_file):
        # Get last line:
        with open(ids_file, "r", encoding="utf-8") as file:
            while line := file.readline():
                prev_line = line
                pass

            category, url = prev_line.split(",")
            if category == last_category:
                return True, category

            return False, category

    return False, None


def ref_ids(mid_url):
    """
    Given an mid_url (i.e., a url for a category of youtube8m videos where youtube video ids are located),
    obtain the corresponding youtube video ids. If not found returns None.
    :param mid_url: URL to get youtube video ids.
    :return: list with ids
    """
    response = requests.get(mid_url)

    if response.status_code == 200:
        text = response.text
        ids = json.loads(text.split(",", 1)[1][:-2])

        return ids

    return None


async def get(url, session, sem):
    """
    Send http GET request in async manner
    :param url: URL to send the request
    :param session: aiohttp.ClientSession
    :param sem: asyncio.Semaphore
    :return: request text and request status
    """
    async with sem:
        async with session.get(url) as response:
            return await response.text(), response.status


async def async_ids(urls):
    """
    Obtain ids in given URLs by using GET HTTP method in async manner. If a given URL returns a status
    different to 200, the id will no be returned since it is not available. This is usually the case
    because either the video was made private or deleted.
    :param urls: URLs from which to get youtube ids.
    :return: list of youtube ids which could be fetched.
    """
    sem = asyncio.Semaphore(50)
    async with aiohttp.ClientSession() as session:
        coroutines = [get(url, session, sem) for url in urls if url]
        aiohttp_responses = await asyncio.gather(*coroutines)

    results = [json.loads(text.split(",")[1][:-2]) for text, status in aiohttp_responses
               if status == 200]
    return results


def gen_ids(mids, categories, batch_size=1000):
    """
    Given a list of mids corresponding to the given categories of youtube8m videos, return a generator
    which provides the id for the next found youtube video id. The function does asynchronous GET requests
    to obtain these ids in batches of the minimum between batch_size and remaining videos for a given category.
    :param mids: list with mids
    :param categories: list with categories
    :param batch_size: size of batches
    :return: list with batches of ids for the current category being looped plus the category
    """
    buffer_urls = deque(maxlen=batch_size)
    mid_urls = [f"{YOUTUBE8M_CATEGORIES_URL}/{mid}.js" for mid in mids]

    for mid_url, category in zip(mid_urls, categories):
        log.info(f"Looping through category {category}")
        reference_ids = ref_ids(mid_url)

        if reference_ids is None:
            log.info("Reference ids for current mid were not found, skipping this category")
            continue

        for i, ref_id in enumerate(reference_ids):
            id_url = f"{YOUTUBE8M_VIDEO_ID_URL}/{ref_id[:2]}/{ref_id}.js"
            buffer_urls.append(id_url)

            if i > 0 and i % batch_size == 0:
                results = asyncio.run(async_ids(buffer_urls))
                buffer_urls.clear()
                yield results, category

        # If category is finished yield the remaining
        results = asyncio.run(async_ids(buffer_urls))
        buffer_urls.clear()
        yield results, category


def save_ids():
    """
    Saves youtube8m ids to specified file in settings
    :return:
    """
    log.info("Saving youtube ids")
    mids, categories = read_mids()
    finished, last_category_saved = is_finished(categories[-1])
    write_mode = "w"

    if finished:
        return

    if last_category_saved:
        mids = list(keep_after(mids, categories, last_category_saved))
        categories = list(keep_after(categories, categories, last_category_saved))
        delete_category(last_category_saved)
        write_mode = "a"

    count = 0
    with open(YOUTUBE8M_IDS_FILE, write_mode, encoding="utf-8") as file:
        for i, values in enumerate(gen_ids(mids, categories)):
            youtube_ids, category = values
            youtube_ids = [category + "," + yid for yid in youtube_ids]
            if i > 0:
                file.write("\n" + "\n".join(youtube_ids))
            else:
                file.write("\n".join(youtube_ids))

            count += len(youtube_ids)
            log.info(f"Saved ids: {count}")

