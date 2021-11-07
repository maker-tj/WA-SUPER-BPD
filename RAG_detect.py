from skimage import graph, data, io, segmentation, color
from matplotlib import pyplot as plt
from skimage.measure import regionprops    # 记录区域特征属性
from skimage import draw
import numpy as np
from skimage.future import graph


def show_img(img):
    width = 10.0
    height = img.shape[0] * width / img.shape[1]  # 高*宽/宽
    f = plt.figure(figsize=(width, height))
    plt.imshow(img)


def _weight_mean_color(graph, src, dst, n):  # 用像素颜色均值做权重值
    # dst:destination image 目标图像
    # scr：source image 源图像
    """Callback to handle merging nodes by recomputing mean color.通过重新计算平均颜色来处理合并节点
    The method expects that the mean color of `dst` is already computed.该方法要求已经计算出“dst”的平均颜色

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.要合并的`图像`中的顶点
    n : int
        A neighbor of `src` or `dst` or both.  “src”或“dst”或两者的邻居

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
        “weight”属性设置为节点“dst”和“n”之间平均颜色的绝对差的字典`
    """
    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)   # 求范数 默认L2范数
    return {'weight': diff}


def merge_mean_color(graph, src, dst):
    # 合并图像块像素颜色均值函数  用像素颜色均值做权重值 方向越相似则权重值越接近，合并权重值接近的图像块
    """Callback called before merging two nodes of a mean color distance graph.在合并平均色距离图的两个节点之前调用回调
    This method computes the mean color of `dst`.此方法计算“dst”的平均颜色

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged. 要合并的`图像`中的顶点
    """
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] / graph.nodes[dst]['pixel count'])


if __name__ == "__main__":
    #img_path = "./tiger.jpg"   # 这种文件路径表示图片在这个superBPD文件夹内
    img_path = "C:/Users/12917/Desktop/detect_images/tiger.jpg"
    img = io.imread(img_path)
    #show_img(img)
    labels = segmentation.slic(img, compactness=30, n_segments=400)  # 先用SLIC分割图像
    labels = labels + 1  # 没有标记的区域是0并且被regionprops忽略
    regions = regionprops(labels)  # 计算每块的多种特征（光照、纹路），用于后面按照特征合并用
    label_rgb = color.label2rgb(labels, img, kind='avg')
    # label2rgb函数为属于一个区域（具有相同标签）的所有像素指定特定颜色.给属于同一类型的像素一样的颜色
    label_rgb = segmentation.mark_boundaries(label_rgb, labels, (0, 0, 0))  # 把进行了slic分割后的分割线画出来
    io.imsave('C:/Users/12917/Desktop/detect_images/1.jpg', label_rgb)
    g = graph.rag_mean_color(img, labels)
    # 计算每个小图块的权重，两个图块越相似权重越接近，这里用图块的像素均值作为每个图块的权重值
    labels2 = graph.merge_hierarchical(labels, g, thresh=35,
                                       rag_copy=False,
                                       in_place_merge=True,
                                       merge_func=merge_mean_color,
                                       weight_func=_weight_mean_color)  # 合并权重相近的图块
    label_rgb2 = color.label2rgb(labels2, img, kind='avg')  # 给属于同一类型的像素一样的颜色
    label_rgb2 = segmentation.mark_boundaries(label_rgb2, labels2, (0, 0, 0))  # 把进行了rag合并以后的分割线（边界）画出来
    io.imsave('C:/Users/12917/Desktop/detect_images/2.jpg', label_rgb2)
