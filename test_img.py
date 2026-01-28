from time import time
from PIL import Image
import numpy as np
import imageio as imio
import os
import torch
import tqdm
import cv2
import matplotlib.pyplot as plt

import viz2d
from lib.spen import SPEN


def refine_matches(spen, d0, d1, idx0, idx1, steerer_ffeat, rot1to2):
    mkpts_0 = d0['keypoints'][0][idx0]
    mkpts_1 = d1['keypoints'][0][idx1]

    f1_patches = d0['feat_patch'][0][idx0]
    f2_patches = d1['feat_patch'][0][idx1]
    
    if steerer_ffeat:
        while(rot1to2>0):
            f1_patches = torch.nn.functional.normalize(steerer(f1_patches))
            rot1to2 = rot1to2 - 1
    pred_mask = spen.net.fine_matcher(f1_patches, f2_patches)

    #Compute fine offsets
    offsets = spen.subpix_softmax2d(pred_mask.squeeze(1))
    mkpts_0 += offsets

    return mkpts_0.cpu().numpy(), mkpts_1.cpu().numpy()


def matching(spen, steerer, im1, im2, min_cossim=-1, steerer_ffeat=False):
    with torch.no_grad():
        if steerer == None:
            matches, kp1, kp2 = spen.match_cmodel(im1, im2, top_k=4096)
            mkpts_0, mkpts_1 = matches[0][:, :2].cpu().numpy(), matches[0][:, 2:].cpu().numpy()
            rot1to2 = 0
        else:
            im_set1 = spen.norm_input(im1)
            im_set2 = spen.norm_input(im2)

            # Compute coarse feats
            out1 = spen.detectAndComputeDense(im_set1, top_k=5000, multiscale=False, type=1)
            out2 = spen.detectAndComputeDense(im_set2, top_k=5000, multiscale=False, type=2)
            out1['descriptors'] = out1['descriptors'][0]
            out2['descriptors'] = out2['descriptors'][0]

            idxs0, idxs1 = spen.match(out1['descriptors'], out2['descriptors'], min_cossim=min_cossim)
            rot1to2 = 0
            for r in range(1, 4):
                # out1['descriptors'] = torch.nn.functional.normalize(steerer(out1['descriptors']), dim=-1)
                out1['descriptors'] = torch.nn.functional.normalize(steerer(out1['descriptors'].unsqueeze(2).unsqueeze(3)).squeeze(3).squeeze(2))
                new_idxs0, new_idxs1 = spen.match(out1['descriptors'], out2['descriptors'], min_cossim=min_cossim)
                if len(new_idxs0) > len(idxs0):
                    idxs0 = new_idxs0
                    idxs1 = new_idxs1
                    rot1to2 = r

            mkpts_0, mkpts_1 = refine_matches(spen, out1, out2, idxs0, idxs1, steerer_ffeat, rot1to2)
            kp1, kp2 = out1['keypoints'], out2['keypoints']

    return mkpts_0, mkpts_1, rot1to2, kp1, kp2


def warp_corners_and_draw_matches(ref_points, dst_points, img1, img2):
    # Calculate the Homography matrix
    H, mask = cv2.findHomography(ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
    mask = mask.flatten()

    # Get corners of the first image (image1)
    h, w = img1.shape[:2]
    corners_img1 = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)

    # Warp corners to the second image (image2) space
    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    # Draw the warped corners in image2
    img2_with_corners = img2.copy()
    for i in range(len(warped_corners)):
        start_point = tuple(warped_corners[i-1][0].astype(int))
        end_point = tuple(warped_corners[i][0].astype(int))
        cv2.line(img2_with_corners, start_point, end_point, (0, 255, 0), 4)  # Using solid green for corners

    # Prepare keypoints and matches for drawMatches function
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
    keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]
    matches = [cv2.DMatch(i,i,0) for i in range(len(mask)) if mask[i]]
    print('Inliner Matches: ', len(matches))

    # Draw inlier matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2_with_corners, keypoints2, matches, None,
                                  matchColor=(0, 255, 0), flags=2)

    return img_matches


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
spen = SPEN(weights='pre_trained/model.pth')
steerer = torch.nn.Conv2d(128, 128, kernel_size=1, padding=0, stride=1, bias=False).to(dev)
steerer.load_state_dict(torch.load('pre_trained/steerer.pth', map_location=dev), strict=True)
steerer.eval()

#Load some example images
im1 = cv2.imread('imgs/opt.png')
im2 = cv2.imread('imgs/sar.png')

#Use out-of-the-box function for extraction + MNN matching
steerer_ffeat = True
tic = time()
mkpts_0, mkpts_1, rot1to2, kp1, kp2 = matching(spen, steerer, im1, im2, min_cossim=0.4, steerer_ffeat=True)
print(f"Number 90 deg rotations from first image to second: {rot1to2}")

canvas = warp_corners_and_draw_matches(mkpts_0, mkpts_1, im1, im2)
toc = time()
print('Running Time: ', toc - tic)
cv2.imwrite('test_result.jpg', canvas)

# viz_keypoint
kp1 = kp1.squeeze(0).cpu().numpy()
kp2 = kp2.squeeze(0).cpu().numpy()

viz2d.plot_images([np.asarray(im1)])
viz2d.plot_keypoints([kp1])
viz_path = 'keypoints1.png'
viz2d.save_plot(viz_path)
print('keypoints1 num: ', len(kp1))

viz2d.plot_images([np.asarray(im2)])
viz2d.plot_keypoints([kp2])
viz_path = 'keypoints2.png'
viz2d.save_plot(viz_path)
print('keypoints2 num: ', len(kp2))