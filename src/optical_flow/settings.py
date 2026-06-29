
YOUTUBE8M_CATEGORIES_URL = "https://storage.googleapis.com/data.yt8m.org/2/j/v"
YOUTUBE8M_VIDEO_ID_URL = "https://storage.googleapis.com/data.yt8m.org/2/j/i"

YOUTUBE8M_IDS_FILE = "config/youtube8m_ids.txt"
YOUTUBE8M_CATEGORIES_FILE ="config/youtube8mcategories.txt"

YOUTUBE_URL = "https://www.youtube.com/watch?v="

## hrnet - for greater detail https://github.com/HRNet/HRNet-Image-Classification (config values are in experiments)
hrnet_w18_cfg = {
    'MODEL': {
        'EXTRA': {
            'STAGE1': {
                'NUM_MODULES': 1,
                'NUM_BRANCHES': 1,
                'BLOCK': 'BOTTLENECK',
                'NUM_BLOCKS': [2],
                'NUM_CHANNELS': [64],
                'FUSE_METHOD': 'SUM'
            },
            'STAGE2': {
                'NUM_MODULES': 1,
                'NUM_BRANCHES': 2,
                'BLOCK': 'BASIC',
                'NUM_BLOCKS': [2, 2],
                'NUM_CHANNELS': [18, 36],
                'FUSE_METHOD': 'SUM'
            },
            'STAGE3': {
                'NUM_MODULES': 3,
                'NUM_BRANCHES': 3,
                'BLOCK': 'BASIC',
                'NUM_BLOCKS': [2, 2, 2],
                'NUM_CHANNELS': [18, 36, 72],
                'FUSE_METHOD': 'SUM'
            },
            'STAGE4': {
                'NUM_MODULES': 2,
                'NUM_BRANCHES': 4,
                'BLOCK': 'BASIC',
                'NUM_BLOCKS': [2, 2, 2, 2],
                'NUM_CHANNELS': [18, 36, 72, 144],
                'FUSE_METHOD': 'SUM'
            }
        }
    }
}