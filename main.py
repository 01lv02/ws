import streamlit as st
from PIL import Image, ImageOps, ImageFilter, ImageDraw, ImageFont
from PIL import Image, ImageEnhance
import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from model import NeuralStyleTransferModel
import settings
import utils

# 创建模型
model = NeuralStyleTransferModel()
st.set_page_config(page_title="图像编辑器", page_icon=":eyeglasses:")


def _compute_content_loss(noise_features, target_features):
    """
    计算指定层上两个特征之间的内容loss
    :param noise_features: 噪声图片在指定层的特征
    :param target_features: 内容图片在指定层的特征
    """
    content_loss = tf.reduce_sum(tf.square(noise_features - target_features))
    # 计算系数
    x = 2. * M * N
    content_loss = content_loss / x
    return content_loss


def compute_content_loss(noise_content_features):
    """
    计算并当前图片的内容loss
    :param noise_content_features: 噪声图片的内容特征
    """
    # 初始化内容损失
    content_losses = []
    # 加权计算内容损失
    for (noise_feature, factor), (target_feature, _) in zip(noise_content_features, target_content_features):
        layer_content_loss = _compute_content_loss(noise_feature, target_feature)
        content_losses.append(layer_content_loss * factor)
    return tf.reduce_sum(content_losses)


def gram_matrix(feature):
    """
    计算给定特征的格拉姆矩阵
    """
    # 先交换维度，把channel维度提到最前面
    x = tf.transpose(feature, perm=[2, 0, 1])
    # reshape，压缩成2d
    x = tf.reshape(x, (x.shape[0], -1))
    # 计算x和x的逆的乘积
    return x @ tf.transpose(x)


def _compute_style_loss(noise_feature, target_feature):
    """
    计算指定层上两个特征之间的风格loss
    :param noise_feature: 噪声图片在指定层的特征
    :param target_feature: 风格图片在指定层的特征
    """
    noise_gram_matrix = gram_matrix(noise_feature)
    style_gram_matrix = gram_matrix(target_feature)
    style_loss = tf.reduce_sum(tf.square(noise_gram_matrix - style_gram_matrix))
    # 计算系数
    x = 4. * (M ** 2) * (N ** 2)
    return style_loss / x


def compute_style_loss(noise_style_features):
    """
    计算并返回图片的风格loss
    :param noise_style_features: 噪声图片的风格特征
    """
    style_losses = []
    for (noise_feature, factor), (target_feature, _) in zip(noise_style_features, target_style_features):
        layer_style_loss = _compute_style_loss(noise_feature, target_feature)
        style_losses.append(layer_style_loss * factor)
    return tf.reduce_sum(style_losses)


def total_loss(noise_features):
    """
    计算总损失
    :param noise_features: 噪声图片特征数据
    """
    content_loss = compute_content_loss(noise_features['content'])
    style_loss = compute_style_loss(noise_features['style'])
    return content_loss * settings.CONTENT_LOSS_FACTOR + style_loss * settings.STYLE_LOSS_FACTOR


# 使用tf.function加速训练
@tf.function
def train_one_step():
    """
    一次迭代过程
    """
    # 求loss
    with tf.GradientTape() as tape:
        noise_outputs = model(noise_image)
        loss = total_loss(noise_outputs)
    # 求梯度
    grad = tape.gradient(loss, noise_image)
    # 梯度下降，更新噪声图片
    optimizer.apply_gradients([(grad, noise_image)])
    return loss


# 创建保存生成图片的文件夹
if not os.path.exists(settings.OUTPUT_DIR):
    os.mkdir(settings.OUTPUT_DIR)

st.image(uploaded_file, channels="BGR")
CONTENT = Image.open(uploaded_file)
CONTENT.save('CONTENT.jpeg')

# 风格转换
style = st.sidebar.selectbox("风格转换", ["原图", "梵高"])

if style == "梵高":
    # 加载内容图片
    content_image = utils.load_images(settings.CONTENT_IMAGE_PATH)
    # 风格图片
    style_image = utils.load_images(settings.STYLE_IMAGE_PATH)

    # 计算出目标内容图片的内容特征备用
    target_content_features = model([content_image, ])['content']
    # 计算目标风格图片的风格特征
    target_style_features = model([style_image, ])['style']

    M = settings.WIDTH * settings.HEIGHT
    N = 3
    # 使用Adma优化器
    optimizer = tf.keras.optimizers.Adam(settings.LEARNING_RATE)

    # 基于内容图片随机生成一张噪声图片
    noise_image = tf.Variable(
        (content_image + np.random.uniform(-0.2, 0.2, (1, settings.HEIGHT, settings.WIDTH, 3))) / 2)

    show = st.empty()
    my_bar = st.progress(0)
    # 共训练settings.EPOCHS个epochs
    for epoch in range(settings.EPOCHS):
        # 使用tqdm提示训练进度
        with tqdm(total=settings.STEPS_PER_EPOCH, desc='Epoch {}/{}'.format(epoch + 1, settings.EPOCHS)) as pbar:
            # 每个epoch训练settings.STEPS_PER_EPOCH次
            for step in range(settings.STEPS_PER_EPOCH):
                _loss = train_one_step()
                pbar.set_postfix({'loss': '%.4f' % float(_loss)})
                pbar.update(1)
            # 每个epoch保存一次图片
            utils.save_image(noise_image, '{}/{}.jpg'.format(settings.OUTPUT_DIR, epoch + 1))
            my_bar.progress(5 * (epoch + 1))
            show.image('output/{}.jpg'.format(epoch + 1))


st.set_page_config(page_title="图像转换编辑器", page_icon=":eyeglasses:")
st.title("图像分类转换器")


# 获取当前文件所在的文件夹路径
path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(path, "export.pkl")

# Load the model
learn_inf = load_learner(model_path)

st.title("Image Classification App")
st.write("Upload an image and the app will predict the corresponding label.")

# Allow the user to upload an image
uploaded_file = st.file_uploader("上传一个图像", type=["png", "jpg", "jpeg"])
if not uploaded_file:
    st.warning("请上传一张图像。")
    st.stop()
# If the user has uploaded an image
if uploaded_file is not None:
    # Display the image
    image = PILImage.create(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Get the predicted label
    pred, pred_idx, probs = learn_inf.predict(image)
    st.write(f"Prediction: {pred}; Probability: {probs[pred_idx]:.04f}")

    # 对上传的图像进行处理和显示
    original_image = Image.open(uploaded_file)
    st.image(original_image, use_column_width=True, caption="原始图像")

    # 对图像进行编辑和处理，省略了调整亮度和对比度等功能
    filtered_image = original_image.filter(ImageFilter.FIND_EDGES)
    cropped_image = filtered_image.crop((100, 100, 400, 400))
    resized_image = cropped_image.resize((224, 224), Image.BICUBIC)
    color_image = ImageOps.posterize(resized_image, 4)

    # 显示最终结果
    st.image(color_image, use_column_width=True, caption="最终结果")
