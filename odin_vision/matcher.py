import pprint
import cv2
from matplotlib import pyplot as plt
import numpy as np

FLANN_INDEX_KDTREE = 1
DIST_MATCHER = 0.7
MIN_MATCH_COUNT = 10


def images_match(
    img1_path, img2_path, min_match_count=MIN_MATCH_COUNT, dist_matcher=DIST_MATCHER
):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    sift = cv2.SIFT_create()  # type: ignore

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # {
    #   algorithm: 1 (FLANN_INDEX_KDTREE),
    #   trees: 5
    # }
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

    # {
    #   checks: 50
    # }
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)  # type: ignore

    # pprint.pprint(index_params)
    # pprint.pprint(search_params)

    matches = flann.knnMatch(des1, des2, k=2)
    # pprint.pprint(matches)

    goods = []
    for m, n in matches:
        if m.distance < dist_matcher * n.distance:
            goods.append(m)

    # pprint.pprint(goods)

    if len(goods) > min_match_count:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in goods]).reshape(-1, 1, 2)  # type: ignore
        dtc_pts = np.float32([kp2[m.trainIdx].pt for m in goods]).reshape(-1, 1, 2)  # type: ignore

        M, mask = cv2.findHomography(src_pts, dtc_pts)
        matchesMask = mask.ravel().tolist()

        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)  # type: ignore
        dst = cv2.perspectiveTransform(pts, M)

        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)  # type: ignore
    else:
        # print("Doesn't match")
        matchesMask = None

    draw_params = dict(
        matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2
    )
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, goods, None, **draw_params)  # type: ignore

    # plt.imshow(img3, "gray")
    # plt.show()

    return [len(goods) > min_match_count, len(goods), img3]
    # cv2.imshow("Correlograma", img3)
    # cv2.waitKey(0)
