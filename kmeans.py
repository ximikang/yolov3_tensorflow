import cv2
import numpy as np

def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    param:
        box: tuple or array, shifted to the origin (i. e. width and height)
        clusters: numpy array of shape (k, 2) where k is the number of clusters
    return:
        numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_

def parse_anno(annotation_file_path):
    """
    parse annotation (labels file)
    :param annotation_file_path: label file path
    :return: array
    """
    anno  = open(annotation_file_path, 'r')
    result = []
    for line in anno:
        s = line.strip().split(' ')
        image = cv2.imread(s[0])
        image_h, image_w = image.shape[:2]
        s = s[1:]
        box_num = len(s) // 5
        for i in range(box_num):
            x_min, y_min, x_max, y_max = float(s[i*5+0]), float(s[i*5+1]), float(s[i*5+2]), float(s[i*5+3])
            width = (x_max - x_min) / image_w
            height = (y_max - y_min) / image_h
            result.append([width, height])
    result = np.asarray(result)
    return result


def kmeans(boxes, k, dist=np.median, seed=1):
    """
    calculates kmeans clustering with iou
    :param boxes:
    :param k: number of clusters
    :param dist: distance function
    :param seed:
    :return:
    """
    rows = boxes.shape[0]
    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed(seed)#random seed=1

    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        # Step 1: allocate each item to the closest cluster centers
        for icluster in range(k):  # I made change to lars76's code here to make the code faster
            distances[:, icluster] = 1 - iou(clusters[icluster], boxes)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        # Step 2: calculate the cluster centers as mean (or median) of all the cases in the clusters.
        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)
        last_clusters = nearest_clusters

    return clusters, nearest_clusters, distances


if __name__ == '__main__':
    train_file_path = "./raccoon_dataset/train.txt"
    anchors_file_path = "./data/raccoon_anchors.txt"
    cluster_num = 9
    anno_result = parse_anno(train_file_path)
    clusters, nearest_clusters, distances = kmeans(anno_result, cluster_num)

    area = clusters[:, 0] * clusters[:, 1]
    indice = np.argsort(area)
    clusters = clusters[indice]
    with open(anchors_file_path, 'w') as f:
        for i in range(cluster_num):
            width, height = clusters[i]
            f.writelines(str(width) + " " + str(height) + " ")
    print("cluster num is: {}".format(cluster_num))


