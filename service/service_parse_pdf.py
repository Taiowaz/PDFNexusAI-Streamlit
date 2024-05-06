import tempfile
import os
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.utils.constants import PartitionStrategy
from unstructured.cleaners.core import clean, group_broken_paragraphs
from unstructured.chunking.title import chunk_by_title
from unstructured .documents.elements import (
    Header,
    Footer,
    Image,
    CompositeElement,
    Table
)

from db_api.api.api_qwenvl import vision_completion

# 分解单个pdf报告


def parse_pdf(filename: str) -> list:
    """ 分割并获取元素 """
    elements = partition_pdf(
        filename=filename,
        strategy=PartitionStrategy.HI_RES,
        infer_table_structure=True,              # 推断表格结构
        extract_images_in_pdf=True,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=False,
        extract_image_block_output_dir=tempfile.gettempdir()
    )

    """  去除页头页脚 """
    filtered_elements = [
        element
        for element in elements
        if not (isinstance(element, Header) or isinstance(element, Footer))
    ]

    """ 对文本进行清洗，图像转换为描述性文本 """
    # 表格在切分时就已经被转换为html
    for element in filtered_elements:
        # 图像转换为描述性文本
        if isinstance(element, Image):
            point1 = element.metadata.coordinates.points[0]
            point2 = element.metadata.coordinates.points[2]
            width = abs(point2[0]-point1[0])
            height = abs(point2[1]-point1[1])
            # 图像清洗
            if width >= 300 and height >= 300:
                element.text = vision_completion(element.metadata.image_path)
        # 非表格元素文本断行合并清洗
        elif not isinstance(element, Table):
            element.text = group_broken_paragraphs(element.text)  # 合并断行
            element.text = clean(
                element.text,
                bullets=False,
                extra_whitespace=True,
                dashes=False,
                trailing_punctuation=False
            )

    """ 进行分块 """
    chunks = chunk_by_title(
        elements=filtered_elements,
        multipage_sections=True,               # 章节跨多页
        combine_text_under_n_chars=128,         # 低于某字符的文本合并
        new_after_n_chars=None,                # 长度超过多少字符后新建
        max_characters=1024,
    )

    text_list = []

    for chunk in chunks:  # 遍历分块，获取文本列表
        # 获取表格html
        if isinstance(chunk, Table):  # 表格转换为html
            if chunk.metadata.text_as_html is not None:
                if text_list:  # 列表不为空
                    text_list[-1] = text_list[-1] + \
                        ":" + chunk.metadata.text_as_html        # 用冒号进行分隔标题与表格，防止与批量转换向量的换行符冲突
                else:
                    text_list.append(chunk.metadata.text_as_html)
        # 文本添加
        elif isinstance(chunk, CompositeElement):
            text = chunk.text
            text_list.append(text)

    # 将文字中的换行符替换为空格，防止与批量转换向量的换行符冲突
    text_list = [text.replace("\n", "//") for text in text_list]

    # 打印日志
    print(f"\nParsed: {filename}\n")

    # 测试代码
    # file_name = filename.split("/")[-1]
    # os.makedirs("file/test/parse_pdf"+file_name, exist_ok=True)
    # i = 0
    # for text in text_list:
    #     with open("file/test/parse_pdf"+file_name+"/"+str(i)+".txt", "w") as f:
    #         f.write(text)
    #     i += 1

    return text_list
