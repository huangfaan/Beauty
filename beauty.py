# coding=utf-8
import numpy as np
from PIL import Image
import cv2
import copy
import time
import os
from main import FaceSegTorch
from easydict import EasyDict as edict


# 遍历目录下的所有图片并返回一个图片信息的列表
def get_dir_files(image_list, dir, ext):
    # print("正在获取文件列表%s"%dir)
    files = os.listdir(dir)
    for filename in files:
        filepath = os.path.join(dir, filename)
        if os.path.isdir(filepath):
            get_dir_files(image_list, filepath, ext)
        elif str.lower(os.path.splitext(filename)[1]) in ext:
            temp = edict()
            temp.filename = filename   # 图片名
            temp.filedir = dir
            temp.filepath = filepath   # 图片路径
            temp.onlyname = os.path.splitext(filename)[0]   # 图片名，不含扩展名
            image_list.append(temp)
        else:
            continue


def add_detail(combined, area_mask):
    if area_mask is not None:
        combined += area_mask
        combined[combined > 1] = 1


def generate_skin(image):
    """genrate skin table"""
    rows, cols, channels = image.shape
    current_img = copy.deepcopy(image)
    # 保边滤波器
    current_img = cv2.edgePreservingFilter(current_img, flags=1, sigma_s=50, sigma_r=0.5)
    skin_table = np.ones(image.shape, np.uint8)
    # Skin is identified by pixels
    for r in range(rows):
        for c in range(cols):
            # get pixel value
            B = current_img.item(r, c, 0)
            G = current_img.item(r, c, 1)
            R = current_img.item(r, c, 2)
            # non-skin area if skin equals 0, skin area equals 1.
            if (abs(R - G) > 15) and (R > G) and (R > B):
                if (R > 95) and (G > 40) and (B > 20) and (max(R, G, B) - min(R, G, B) > 15):
                    pass
                elif (R > 220) and (G > 210) and (B > 170):
                    pass
                else:
                    skin_table.itemset((r, c, 2), 0)
                    skin_table.itemset((r, c, 1), 0)
                    skin_table.itemset((r, c, 0), 0)
            else:
                skin_table.itemset((r, c, 2), 0)
                skin_table.itemset((r, c, 1), 0)
                skin_table.itemset((r, c, 0), 0)
    return skin_table


def generate_skin_by_seg(image, seg_detail):
    """genrate skin table"""
    skin_table = np.zeros([image.shape[0], image.shape[1]], dtype=np.uint8)
    add_detail(skin_table, seg_detail["skin"])
    add_detail(skin_table, seg_detail["l_brow"])
    add_detail(skin_table, seg_detail["r_brow"])
    add_detail(skin_table, seg_detail["nose"])
    add_detail(skin_table, seg_detail["u_lip"])
    add_detail(skin_table, seg_detail["l_lip"])
    add_detail(skin_table, seg_detail["mouth"])
    add_detail(skin_table, seg_detail["l_ear"])
    add_detail(skin_table, seg_detail["r_ear"])

    return np.tile(np.expand_dims(skin_table, axis=2), (1, 1, 3))



def add_edge_detail(image, buffer_image, p, w):
    '''
    高反差：Src - GuassBlur(Src)
    Dest = 高反差 * p + (Src + 2 * GuassBlur(EPFFilter(Src) - Src + 128) - 256) ;
    '''
    # 得到图像的高反差
    gauss = cv2.GaussianBlur(image, ksize=(5, 5), sigmaX=0)
    highPass = cv2.subtract(image, gauss)

    contr = cv2.subtract(buffer_image, image)
    contr = contr + 128
    gaus = cv2.GaussianBlur(contr, ksize=(3, 3), sigmaX=0)
    # cv2.imshow('3', gaus)
    # cv2.waitKey(0)

    gaus = gaus.astype(np.float32)
    temp = gaus + gaus + image - 256
    dst = highPass * p + temp
    dst[dst > 255] = 255
    dst = dst.astype(np.uint8)
    # dst = cv2.addWeighted(dst, 1 - w, image, w, 0.0)  # 质感
    return dst



def add_edge_detail2(image, buffer_image, w):
    '''
    Dest =Src * (1-w) + (Src + 2 * GuassBlur(EPFFilter(Src) - Src + 128) - 256) * w;
    '''
    contr = cv2.subtract(buffer_image, image)
    contr = cv2.add(contr, (10, 10, 10, 128))
    gaus = cv2.GaussianBlur(contr, ksize=(3, 3), sigmaX=0)
    B = cv2.subtract(cv2.add(cv2.add(gaus, gaus), image), (10, 10, 10, 255))
    dst = cv2.addWeighted(image, w, B, 1 - w, 0.0)
    return dst


# 导向滤波对于1024*1024图像 大概在120-150ms之间
def guideFilter(I, p, r, eps):
    mean_I = cv2.blur(I, (r, r))
    mean_p = cv2.blur(p, (r, r))
    corr_I = cv2.blur(I*I, (r, r))
    corr_Ip = cv2.blur(I*p, (r, r))
    var_I = corr_I - mean_I * mean_I  # 方差
    cov_Ip = corr_Ip - mean_I * mean_p  # 协方差

    a = cov_Ip / (var_I+ eps)
    b = mean_p - a * mean_I
    mean_a = cv2.blur(a, (r, r))
    mean_b = cv2.blur(b, (r, r))
    q = mean_a * I + mean_b

    return q


# 快速导向滤波  对于1024*1024图像 大约30ms
def fastguideFilter(I, p, r, eps, s):
    # 输入图像的高、宽
    h, w = I.shape[:2]

    # 缩小图像
    size = (int(round(w * s)), int(round(h * s)))
    small_I = cv2.resize(I, size, interpolation=cv2.INTER_CUBIC)
    small_p = cv2.resize(p, size, interpolation=cv2.INTER_CUBIC)

    # 缩小滑动窗口
    small_winSize = (int(round(r * s)), int(round(r * s)))

    # I的均值平滑 p的均值平滑
    mean_small_I = cv2.blur(small_I, small_winSize)
    mean_small_p = cv2.blur(small_p, small_winSize)

    # I*I和I*p的均值平滑
    mean_small_II = cv2.blur(small_I * small_I, small_winSize)
    mean_small_Ip = cv2.blur(small_I * small_p, small_winSize)

    # 方差、协方差
    var_small_I = mean_small_II - mean_small_I * mean_small_I
    cov_small_Ip = mean_small_Ip - mean_small_I * mean_small_p

    small_a = cov_small_Ip / (var_small_I + eps)
    small_b = mean_small_p - small_a * mean_small_I

    # 对a、b进行均值平滑
    mean_small_a = cv2.blur(small_a, small_winSize)
    mean_small_b = cv2.blur(small_b, small_winSize)

    # 放大
    size1 = (w, h)
    mean_a = cv2.resize(mean_small_a, size1, interpolation=cv2.INTER_LINEAR)
    mean_b = cv2.resize(mean_small_b, size1, interpolation=cv2.INTER_LINEAR)

    q = mean_a * I + mean_b

    return q


def buffing(image, skin_table, r=12, eps=0.1*0.1, s=0.2):  # 100msz左右
    kernel = np.array((
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625]), dtype="float32")
    skin_table = cv2.filter2D(skin_table, -1, kernel)  # 为了实现图像融合  10ms


    # buffing  30ms
    I = np.float32(image)
    p = I
    # current_img = guideFilter(I, p, r, eps*255*255)  # 注意：若输入图像归一化  则esp就不需要乘以255*255
    current_img = fastguideFilter(I, p, r, eps*255*255, s)
    # current_img = cv2.edgePreservingFilter(image, flags=1, sigma_s=50, sigma_r=grade * 0.05) # grade 取 5
    current_img[current_img > 255] = 255

    imgskin_c = 1 - skin_table
    skin = current_img * skin_table  # 10ms
    res = np.uint8(image * imgskin_c + skin)  # 10ms
    # skin = white_skin(skin, 0)  # 磨皮中插入美白算法8，值为0时表示没添加美白
    return res


def buffing_bb(image, skin_table, grade=5):
    kernel = np.array((
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625]), dtype="float32")
    skin_table = cv2.filter2D(skin_table, -1, kernel)  # 为了实现图像融合

    # buffing
    value = grade * 0.05
    current_img = cv2.edgePreservingFilter(image, flags=1, sigma_s=50, sigma_r=value)
    imgskin_c = 1 - skin_table  # np.uint8(1 - skin_table)
    skin = current_img * skin_table
    # skin = white_skin(skin, 0)  # 磨皮中插入美白算法8，  值为0时表示没添加美白
    return np.uint8(image * imgskin_c + skin)


# 肤色美白, power : 0 ~ 100, 值为0时表示没添加美白
def white_skin(skin, power=5):
    """
    power : 0 ~ 100
    """
    power = power / 100.0
    rows, cols, channels = skin.shape
    for r in range(rows):
        for c in range(cols):
            B = skin.item(r, c, 0)
            G = skin.item(r, c, 1)
            R = skin.item(r, c, 2)
            max_value = max([R, G, B])
            degree = 255 - max_value - int(max_value * power)
            if degree >= 0:
                degree = int(max_value * power)
            else:
                degree = 255 - max_value
            skin.itemset((r, c, 0), B + degree)
            skin.itemset((r, c, 1), G + degree)
            skin.itemset((r, c, 2), R + degree)
    return skin


if __name__ == "__main__":
    image_dir = './zipai_final_tex'
    out_dir = './zipai_out'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    image_list = []
    get_dir_files(image_list, image_dir, ['.jpg', '.png'])

    FaceSegTorch.warm_up()  # 初始化模型 用于皮肤检测

    for idx, item in enumerate(image_list):
        print(idx)
        image_path = item['filepath']
        image = cv2.imread(image_path)

        res, seg_detail = FaceSegTorch.face_seg(image)
        # skin_table = generate_skin(image)
        skin_table = generate_skin_by_seg(image, seg_detail)

        buffer_image = buffing(image, skin_table, r=10, eps=0.002, s=2)  # 50-60ms左右
        dst = add_edge_detail(image, buffer_image, 0.8, 0)  # 30ms
        dst2 = add_edge_detail2(image, buffer_image, 0.1)  # 10ms

        bb = buffing_bb(image, skin_table, 5)
        dst_bb = add_edge_detail(image, buffer_image, 0.8, 0)
        dst2_bb = add_edge_detail2(image, buffer_image, 0.1)


        h, w, c = image.shape
        result = np.zeros((h * 2, w * 3, c), image.dtype)
        result[:h, :w, :] = buffer_image
        result[:h, w:w*2, :] = dst
        result[:h, w*2:, :] = dst2
        result[h:, :w, :] = bb
        result[h:, w:w*2, :] = dst_bb
        result[h:, w*2:, :] = dst2_bb
        cv2.imwrite(os.path.join(out_dir, item['filename']), dst)

