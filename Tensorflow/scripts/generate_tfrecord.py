"""
Usage:
  # Create train data:
  python generate_tfrecord.py -x [PATH_TO_IMAGES]/train -l [PATH_TO_ANNOTATIONS]/label_map.pbtxt -o [PATH_TO_ANNOTATIONS]/train.record

  # Create test data:
  python generate_tfrecord.py -x [PATH_TO_IMAGES]/test -l [PATH_TO_ANNOTATIONS]/label_map.pbtxt -o [PATH_TO_ANNOTATIONS]/test.record
"""

import os
import glob
import pandas as pd
import io
import xml.etree.ElementTree as ET

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple

import argparse

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    return pd.DataFrame(xml_list, columns=column_name)

def class_text_to_int(row_label, label_map_dict):
    return label_map_dict.get(row_label, None)

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def create_tf_example(group, path, label_map_dict):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    image = Image.open(io.BytesIO(encoded_jpg))
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'  # or b'png'

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class'], label_map_dict))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def load_label_map(label_map_path):
    label_map_dict = {}
    with open(label_map_path, 'r') as f:
        for line in f:
            if "name:" in line:
                name = line.strip().split(":")[1].strip().strip("'").strip('"')
            if "id:" in line:
                idx = int(line.strip().split(":")[1])
                label_map_dict[name] = idx
    return label_map_dict

if __name__ == '__main__':
    import tensorflow as tf

    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--image_dir', help='Path to images directory', required=True)
    parser.add_argument('-l', '--label_map', help='Path to label_map.pbtxt file', required=True)
    parser.add_argument('-o', '--output_path', help='Path to output TFRecord file', required=True)
    args = parser.parse_args()

    label_map_dict = load_label_map(args.label_map)
    examples = xml_to_csv(args.image_dir)
    grouped = split(examples, 'filename')
    writer = tf.io.TFRecordWriter(args.output_path)

    for group in grouped:
        tf_example = create_tf_example(group, args.image_dir, label_map_dict)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print('Successfully created the TFRecord file: {}'.format(args.output_path))
