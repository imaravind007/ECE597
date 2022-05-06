import tensorflow as tf

from src.com_in_ineuron_ai_detectfaces_mtcnn.mtcnn2 import PNet, RNet, ONet
# from src.com_in_ineuron_ai_tracker_utils.tools import detect_face, get_model_filenames


# class FaceDetectorMtcnN(threading.Thread):
class FaceDetectorMtcnN():
    def __init__(self, args):
        # threading.Thread.__init__(self)
        self.file_paths = get_model_filenames(args.model_dir)
        self.minsize = args.minsize
        self.threshold = args.thresholdVal
        self.factor = args.factor
        self.save_image = args.save_image
        self.save_name = args.save_name
        # self.img_path = args.image_path

    def run(self, img):
        # img = cv2.imread(self.img_path)
        with tf.device('/gpu:0'):
            with tf.Graph().as_default():
                config = tf.ConfigProto(allow_soft_placement=True)
                with tf.Session(config=config) as sess:
                    if len(self.file_paths) == 3:
                        image_pnet = tf.placeholder(
                            tf.float32, [None, None, None, 3])
                        pnet = PNet({'data': image_pnet}, mode='test')
                        out_tensor_pnet = pnet.get_all_output()

                        image_rnet = tf.placeholder(tf.float32, [None, 24, 24, 3])
                        rnet = RNet({'data': image_rnet}, mode='test')
                        out_tensor_rnet = rnet.get_all_output()

                        image_onet = tf.placeholder(tf.float32, [None, 48, 48, 3])
                        onet = ONet({'data': image_onet}, mode='test')
                        out_tensor_onet = onet.get_all_output()

                        saver_pnet = tf.train.Saver(
                                        [v for v in tf.global_variables()
                                         if v.name[0:5] == "pnet/"])
                        saver_rnet = tf.train.Saver(
                                        [v for v in tf.global_variables()
                                         if v.name[0:5] == "rnet/"])
                        saver_onet = tf.train.Saver(
                                        [v for v in tf.global_variables()
                                         if v.name[0:5] == "onet/"])

                        saver_pnet.restore(sess, self.file_paths[0])

                        def pnet_fun(img): return sess.run(
                            out_tensor_pnet, feed_dict={image_pnet: img})

                        saver_rnet.restore(sess, self.file_paths[1])

                        def rnet_fun(img): return sess.run(
                            out_tensor_rnet, feed_dict={image_rnet: img})

                        saver_onet.restore(sess, self.file_paths[2])

                        def onet_fun(img): return sess.run(
                            out_tensor_onet, feed_dict={image_onet: img})

                    else:
                        saver = tf.train.import_meta_graph(self.file_paths[0])
                        saver.restore(sess, self.file_paths[1])

                        def pnet_fun(img): return sess.run(
                            ('softmax/Reshape_1:0',
                             'pnet/conv4-2/BiasAdd:0'),
                            feed_dict={
                                'Placeholder:0': img})

                        def rnet_fun(img): return sess.run(
                            ('softmax_1/softmax:0',
                             'rnet/conv5-2/rnet/conv5-2:0'),
                            feed_dict={
                                'Placeholder_1:0': img})

                        def onet_fun(img): return sess.run(
                            ('softmax_2/softmax:0',
                             'onet/conv6-2/onet/conv6-2:0',
                             'onet/conv6-3/onet/conv6-3:0'),
                            feed_dict={
                                'Placeholder_2:0': img})

                    # start_time = time.time()
                    bounding_boxes = detect_face(img, self.minsize,
                                                     pnet_fun, rnet_fun, onet_fun,
                                                     self.threshold, self.factor)
        return bounding_boxes
