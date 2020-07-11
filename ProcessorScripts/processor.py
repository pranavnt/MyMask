# Portions of this code are based on the example code here:
# https://github.com/microsoft/Deep3DFaceReconstruction/blob/master/demo.py

import sys, face_alignment, random
from skimage import io
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import glob
from scipy.io import loadmat, savemat

from preprocess_img import Preprocess
from load_data import *
from face_decoder import Face3D


def main(argv):
    input_img = argv[0]

    # Generate the text file for face alignment
    face_alignment_data = get_alignment(input_img)
    alignment_data_file_path = change_file_extension(input_img, "txt")
    with open(alignment_data_file_path, 'w') as file:
        file.write(face_alignment_data)

    # Now send that to Deep3DFaceReconstruction
    d3dr(input_img)


def change_file_extension(path, new_ext):
    return ".".join(path.split(".")[:-1]) + "." + new_ext


def d3dr(file):
    # read BFM face model
    # transfer original BFM model to our model
    if not os.path.isfile('./BFM/BFM_model_front.mat'):
        transferBFM09()

    # read standard landmarks for preprocessing images
    lm3D = load_lm3d()
    batchsize = 1
    n = 0

    # build reconstruction model
    with tf.Graph().as_default() as graph, tf.device('/cpu:0'):
        FaceReconstructor = Face3D()
        images = tf.placeholder(name='input_imgs', shape=[batchsize, 224, 224, 3], dtype=tf.float32)
        graph_def = load_graph('network/FaceReconModel.pb')
        tf.import_graph_def(graph_def, name='resnet', input_map={'input_imgs:0': images})

        # output coefficients of R-Net (dim = 257)
        coeff = graph.get_tensor_by_name('resnet/coeff:0')

        # reconstructing faces
        FaceReconstructor.Reconstruction_Block(coeff, batchsize)
        face_shape = FaceReconstructor.face_shape_t
        face_texture = FaceReconstructor.face_texture
        face_color = FaceReconstructor.face_color
        landmarks_2d = FaceReconstructor.landmark_p
        recon_img = FaceReconstructor.render_imgs
        tri = FaceReconstructor.facemodel.face_buf

        with tf.Session() as sess:
            print('reconstructing...')
            n += 1
            print(n)
            # load images and corresponding 5 facial landmarks
            img, lm = load_img(file, file.replace('png', 'txt').replace('jpg', 'txt'))
            # preprocess input image
            input_img, lm_new, transform_params = Preprocess(img, lm, lm3D)

            coeff_, face_shape_, face_texture_, face_color_, landmarks_2d_, recon_img_, tri_ = sess.run([coeff,
                                                                                                         face_shape,
                                                                                                         face_texture,
                                                                                                         face_color,
                                                                                                         landmarks_2d,
                                                                                                         recon_img,
                                                                                                         tri],
                                                                                                        feed_dict={
                                                                                                            images: input_img})

            # reshape outputs
            input_img = np.squeeze(input_img)
            face_shape_ = np.squeeze(face_shape_, (0))
            face_texture_ = np.squeeze(face_texture_, (0))
            face_color_ = np.squeeze(face_color_, (0))
            landmarks_2d_ = np.squeeze(landmarks_2d_, (0))
            recon_img_ = np.squeeze(recon_img_, (0))

            # save output files
            savemat(change_file_extension(file, "mat"),
                    {'cropped_img': input_img[:, :, ::-1], 'recon_img': recon_img_, 'coeff': coeff_,
                     'face_shape': face_shape_, 'face_texture': face_texture_, 'face_color': face_color_,
                     'lm_68p': landmarks_2d_, 'lm_5p': lm_new})
            save_obj(change_file_extension(file, "obj"),
                     face_shape_, tri_,
                     np.clip(face_color_, 0, 255) / 255)  # 3D reconstruction face (in canonical view)


def load_graph(graph_filename):
    with tf.gfile.GFile(graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    return graph_def


def get_alignment(input_img):
    # We don't have a server with a GPU so this has to run on CPU
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device="cpu")
    input = io.imread(input_img)
    output = list(fa.get_landmarks(input)[-1])
    output = random.sample(output, 5)

    # Convert that into a TSV
    tsv_output = ""
    for data_point in output:
        tsv_output += str(data_point[0])
        tsv_output += "\t"
        tsv_output += str(data_point[1])
        tsv_output += "\n"

    return tsv_output.rstrip()


if __name__ == "__main__":
    main(sys.argv[1:])
