import numpy as np
import cv2
import os
import tensorflow as tf
import tqdm
import utils

from remove_flare import BlareRemoval

def eval_test():
    input_dir = "/home/sivtsovdt/arcadia/ads/pytorch/embedding_model/Improving-Lens-Flare-Removal/test_images"

    blare_model = BlareRemoval()

    # input_files = sorted(tf.io.gfile.glob(os.path.join(input_dir, '*.*g')))
    input_files = ["/home/sivtsovdt/arcadia/ads/pytorch/embedding_model/Improving-Lens-Flare-Removal/test_images/Upscales.ai_1703287188661.jpeg",]
    for input_file in tqdm.tqdm(input_files):
        with tf.io.gfile.GFile(input_file, 'rb') as f:
            blob = f.read()
        input_u8 = tf.image.decode_image(blob)[Ellipsis, :3]

        out = blare_model.Process(input_u8)[2]

        path = "/home/sivtsovdt/arcadia/ads/pytorch/embedding_model/Improving-Lens-Flare-Removal/AAAAAA_out.png"
        utils.write_image(out, path)

        # cv2.imwrite("/home/sivtsovdt/arcadia/ads/pytorch/embedding_model/Improving-Lens-Flare-Removal/AAAAAA_in.png", im)
        # cv2.imwrite("/home/sivtsovdt/arcadia/ads/pytorch/embedding_model/Improving-Lens-Flare-Removal/AAAAAA_out.png", out)
        # with tf.io.gfile.GFile("/home/sivtsovdt/arcadia/ads/pytorch/embedding_model/Improving-Lens-Flare-Removal/AAAAAA_out.png", 'wb') as f:
        #     f.write(out)

        # path = "/home/sivtsovdt/arcadia/ads/pytorch/embedding_model/Improving-Lens-Flare-Removal/AAAAAA_out.png"
        # image_u8 = tf.image.convert_image_dtype(out, tf.uint8, saturate=True)
        # if path.lower().endswith('.png'):
        #     encoded = tf.io.encode_png(image_u8)
        # elif path.lower().endswith('.jpg') or path.lower().endswith('.jpeg'):
        #     encoded = tf.io.encode_jpeg(image_u8, progressive=True)
        # with tf.io.gfile.GFile(path, 'wb') as f:
        #     f.write(encoded.numpy())
        break

if __name__=="__main__":
    eval_test()