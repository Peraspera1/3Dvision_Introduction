import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math 
# 读取图像
def load_image(image_path):
    # 使用Pillow打开图片并将其转换为RGB格式
    image = Image.open(image_path).convert('RGB')
    # 转换为numpy数组
    image_np = np.array(image)
    return image_np

# 亮度检测函数
def detect_brightness(image_np, threshold=128):
    # 将图像从RGB转换为灰度
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # 应用阈值, 阈值以上的设置为255(白)，以下设置为0(黑)
    _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    
    return gray_image, binary_image

# 显示原图、灰度图和二值化图像
def display_images(original_image, gray_image, binary_image):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(binary_image, cmap='gray')
    plt.title('Thresholded Brightness')
    plt.axis('off')
    
    plt.show()

# 查找三角形的顶点
def find_triangle_vertices(binary_image):
    # 查找轮廓
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 假设我们只有一个三角形，取第一个轮廓
    contour = contours[0]
    
    # 计算凸包（Convex Hull），这会给出三角形的外部轮廓
    hull = cv2.convexHull(contour)
    
    # 凸包中的点即为三角形的顶点
    vertices = []
    for point in hull:
        vertices.append(tuple(point[0]))  # 取出x, y坐标
    
    return vertices

# 显示顶点在图像上的位置
def display_vertices(binary_image, vertices):
    # 将顶点绘制在原图上
    vertex_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)
    
    for vertex in vertices:
        cv2.circle(vertex_image, vertex, 5, (255, 0, 0), -1)  # 用蓝色标记顶点，半径为5
    
    # 显示图像
    plt.imshow(vertex_image)
    plt.title('Triangle with Vertices')
    plt.axis('off')
    plt.show()

# 查找纵坐标最大的顶点
def find_highest_vertex(vertices):
    # 假设第一个顶点的y值最大
    highest_vertex = vertices[0]
    
    # 遍历所有顶点，找到y坐标最大的点
    for vertex in vertices:
        if vertex[0] > highest_vertex[0]:  # vertex[1]是y坐标
            highest_vertex = vertex
    
    return highest_vertex

def find_lowest_vertex(vertices):
    # 假设第一个顶点的y值最小
    lowest_vertex = vertices[0]
    
    # 遍历所有顶点，找到y坐标最小的点
    for vertex in vertices:
        if vertex[0] < lowest_vertex[0]:  # vertex[1]是y坐标
            lowest_vertex = vertex
    
    return lowest_vertex


# 查找纵坐标最大的顶点
def find_highest_vertex2(vertices):
    # 假设第一个顶点的y值最大
    highest_vertex = vertices[0]
    
    # 遍历所有顶点，找到y坐标最大的点
    for vertex in vertices:
        if vertex[1] < highest_vertex[1]:  # vertex[1]是y坐标
            highest_vertex = vertex
    
    return highest_vertex

def cal_angle(b_z, t_z):
    return 2 * math.atan(0.5 * b_z / t_z)

if __name__ == "__main__":
    # 图像文件路径
    image_path = 'C:/Users/admin/Desktop/a1/111.bmp' # 确保这是一个图片文件路径
    
    # 加载图像
    image_np = load_image(image_path)
    
    # 亮度检测并应用阈值调整
    gray_image, binary_image = detect_brightness(image_np, threshold=40)  # 可以调整threshold的值


    # 查找三角形的顶点
    vertices = find_triangle_vertices(binary_image)
    print("Triangle vertices:", vertices)
    vertice_1 = find_highest_vertex(vertices)
    # 输出顶点坐标
    print("Triangle vertices:", vertice_1)
    
    vertice_2 = find_lowest_vertex(vertices)
    # 输出顶点坐标
    print("Triangle vertices:", vertice_2)

    vertice_3 = find_highest_vertex2(vertices)
    print("Triangle vertices:", vertice_3)

    bottom = vertice_1[0] - vertice_2[0]
    b_z = 0.5*(vertice_1[1] + vertice_2[1])
    t_z = b_z - vertice_3[1]

    angle = cal_angle(bottom, t_z)
    print("三角形底边：", bottom, "三角形高", t_z, "顶角", angle)

    # 显示顶点
    display_vertices(binary_image, vertices)
    # 显示原始图像、灰度图像和阈值后的二值化图像
    display_images(image_np, gray_image, binary_image)
