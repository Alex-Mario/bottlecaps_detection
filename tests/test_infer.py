from scr.bsort.infer import infer_image

def test_infer_image():
    config = {
        "inference": {"conf_threshold": 0.25}
    }
    # gunakan sample image dari dataset
    infer_image("test/images/raw-250110_dc_s001_b3_4_jpg.rf.582eba58713c7f949249b6d32ac0c312.jpg", config)