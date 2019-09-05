import os
import numpy as np
import tensorflow as tf
from forward_kinematics import FK
from ops import qlinear, q_mul_q, conv1d, lrelu, get_wjs
from tensorflow import atan2
from tensorflow import asin


class pmnet_model(object):
    def __init__(self,
                 batch_size,
                 alpha,
                 gamma,
                 omega,
                 euler_ord,
                 n_joints,
                 max_len,
                 parents,
                 keep_prob,
                 learning_rate,
                 optim_name,
                 local_mean,
                 local_std,
                 global_mean,
                 global_std,
                 logs_dir,
                 margin,
                 norm_type,
                 GAN_Balancing,
                 is_train=True
                 ):

        self.n_joints = n_joints
        self.batch_size = batch_size
        self.alpha = alpha  # threshold for euler angle
        self.gamma = gamma  # weight factor for twist loss
        self.omega = omega  # weight factor for smooth loss
        self.euler_ord = euler_ord
        self.kp = keep_prob
        self.max_len = max_len
        self.learning_rate = learning_rate
        self.fk = FK()

        self.margin = margin
        self.fake_score = 0.5
        self.norm_type = norm_type
        self.balancing = GAN_Balancing
        self.kp_ = tf.placeholder(tf.float32, shape=[], name="kp")

        self.seqA = tf.placeholder(tf.float32, shape=[batch_size, max_len, 3 * n_joints + 4], name="seqA")
        self.seqB = tf.placeholder(tf.float32, shape=[batch_size, max_len, 3 * n_joints + 4], name="seqB")
        self.skelA = tf.placeholder(tf.float32, shape=[batch_size, max_len, 3 * n_joints], name="skelA")
        self.skelB = tf.placeholder(tf.float32, shape=[batch_size, max_len, 3 * n_joints], name="skelB")
        self.mask = tf.placeholder(tf.float32, shape=[batch_size, max_len], name="mask")
        self.aeReg = tf.placeholder(tf.float32, shape=[batch_size, 1], name="aeReg")

        self.inp_height = tf.placeholder(tf.float32, shape=[batch_size, 1], name="inp_height")
        self.tgt_height = tf.placeholder(tf.float32, shape=[batch_size, 1], name="tgt_height")

        seqlen = tf.reduce_sum(self.mask, axis=1)
        seqlen = tf.cast(seqlen, tf.int32)

        """ Inverse kinematics (IK) result (Reconstruction) """
        a_locals_ik = []
        a_quats_ik = []

        """ delta q """
        dqs = []

        """ Retargeting result """
        b_locals_rt = []
        b_quats_rt = []

        """ Reference poses for the input and the target """
        tpose_a = tf.reshape(self.skelA[:, 0, :], [batch_size, n_joints, 3])
        tpose_a = tpose_a * local_std + local_mean
        self.refA = tpose_a

        tpose_b = tf.reshape(self.skelB[:, 0, :], [batch_size, n_joints, 3])
        tpose_b = tpose_b * local_std + local_mean
        self.refB = tpose_b
        self.refB_feed = self.skelB[:, 0, :]

        a_pose_features = []
        b_pose_features = []

        # ========================================================================== #
        # ========================= Mapping pose from A to B ======================= #
        # ========================================================================== #
        with tf.variable_scope("Local_Motion"):
            reuse = False
            for t in range(max_len):
                """ Pose Encoder for A"""
                inputA_t = self.seqA[:, t, :-4]  # only using locals

                hidden1 = tf.layers.dense(inputA_t, 512, activation=tf.nn.relu, name='Backbone_h1', reuse=reuse)
                hidden1 = tf.layers.dropout(hidden1, 1 - self.kp, training=is_train)

                hidden2 = tf.layers.dense(hidden1, 512, activation=tf.nn.relu, name='Backbone_h2', reuse=reuse)
                hidden2 = tf.layers.dropout(hidden2, 1 - self.kp, training=is_train)

                hidden3 = tf.layers.dense(hidden2, 512, activation=tf.nn.relu, name='Backbone_h3', reuse=reuse)
                hidden3 = tf.layers.dropout(hidden3, 1 - self.kp, training=is_train)

                hidden4 = tf.layers.dense(hidden3, 512, activation=tf.nn.relu, name='Backbone_h4', reuse=reuse)

                a_pose_features.append(hidden4)

                """ Inverse Kinematics for A """
                qoutA_t = qlinear(hidden4, 4 * self.n_joints, name='IKnet', reuse=reuse)
                qoutA_t = tf.reshape(qoutA_t, [batch_size, n_joints, 4])
                qoutA_t = self.normalized(qoutA_t)
                a_quats_ik.append(qoutA_t)

                """ Mapping A to B (edited on 04/21) """
                ref_embed = tf.layers.dense(self.refB_feed, 16, activation=tf.nn.sigmoid,
                                            name='Ref_embedding', reuse=reuse)
                ref_embed = tf.layers.dropout(ref_embed, 1 - self.kp, training=is_train)

                combined_in = tf.concat(values=[hidden4, ref_embed], axis=-1)

                dq_t = qlinear(combined_in, 4 * self.n_joints, name="Mapping", reuse=reuse)
                dq_t = tf.reshape(dq_t, [batch_size, n_joints, 4])
                dq_t = self.normalized(dq_t)
                dqs.append(dq_t)

                """ Hamilton product """
                qb_rt = q_mul_q(qoutA_t, dq_t)
                b_quats_rt.append(qb_rt)

                """ Forward Kinematics for B """
                localB_out_t = self.fk.run(parents, self.refB, qb_rt)
                localB_out_t = (localB_out_t - local_mean) / local_std
                b_locals_rt.append(localB_out_t)

                if is_train:
                    """ Reconstruct A """
                    localA_out_t = self.fk.run(parents, self.refA, qoutA_t)
                    localA_out_t = (localA_out_t - local_mean) / local_std  # shape: (batch_size, 22, 3)
                    a_locals_ik.append(localA_out_t)

                    """ Pose Encoder again for B """
                    inputB_t = b_locals_rt[-1]
                    inputB_t = tf.reshape(inputB_t, [batch_size, -1])

                    hidden1 = tf.layers.dense(inputB_t, 512, activation=tf.nn.relu, name='Backbone_h1', reuse=True)
                    hidden1 = tf.layers.dropout(hidden1, 1 - self.kp, training=is_train)

                    hidden2 = tf.layers.dense(hidden1, 512, activation=tf.nn.relu, name='Backbone_h2', reuse=True)
                    hidden2 = tf.layers.dropout(hidden2, 1 - self.kp, training=is_train)

                    hidden3 = tf.layers.dense(hidden2, 512, activation=tf.nn.relu, name='Backbone_h3', reuse=True)
                    hidden3 = tf.layers.dropout(hidden3, 1 - self.kp, training=is_train)

                    hidden4 = tf.layers.dense(hidden3, 512, activation=tf.nn.relu, name='Backbone_h4', reuse=True)

                    b_pose_features.append(hidden4)

                reuse = True

        """ Feed-forward Local results """
        self.quatA_ik = tf.stack(a_quats_ik, axis=1)    # shape: (batch_size, max_len, 22, 4)
        self.quatB_rt = tf.stack(b_quats_rt, axis=1)    # shape: (batch_size, max_len, 22, 4)
        self.deltaQ = tf.stack(dqs, axis=1)
        self.localB_rt = tf.stack(b_locals_rt, axis=1)  # shape: (batch_size, max_len, 22, 3)

        # ========================================================================== #
        # ================== Mapping overall movements from A to B ================= #
        # ========================================================================== #
        ga_vel = self.seqA[:, :, -4:-1]
        ga_rot = self.seqA[:, :, -1]

        """ Normalize (edited on 04/16) """
        self.normalized_vin = tf.concat((tf.divide(ga_vel, self.inp_height[:, :, None]),
                                         ga_rot[:, :, None]), axis=-1)

        """ Movement Regressor """
        reuse = False
        with tf.variable_scope("Global_Motion", reuse=reuse):
            h0 = lrelu(conv1d(self.normalized_vin, 128, k_w=3, d_w=1, name="GM_h0"))
            vb_out = conv1d(h0, 4, k_w=3, d_w=1, name="GM_h1")

        self.normalized_vout = vb_out

        """ De-normalize """
        gb_vel = self.normalized_vout[:, :, :-1]
        gb_rot = self.normalized_vout[:, :, -1]
        self.globalB_rt = tf.concat((tf.multiply(gb_vel, self.tgt_height[:, :, None]),
                                     gb_rot[:, :, None]), axis=-1)

        if is_train:
            self.localA_ik = tf.stack(a_locals_ik, axis=1)

            localA_gt = tf.reshape(self.seqA[:, :, :-4], [batch_size, max_len, n_joints, 3])
            localB_gt = tf.reshape(self.seqB[:, :, :-4], [batch_size, max_len, n_joints, 3])

            globalA_gt = self.seqA[:, :, -4:]

            self.posefeatureA = tf.stack(a_pose_features, axis=1)  # shape: (batch_size, max_len, 512)
            self.posefeatureB = tf.stack(b_pose_features, axis=1)  # shape: (batch_size, max_len, 512)

            # ========================================================================== #
            # =========================== Motion Discriminator ========================= #
            # ========================================================================== #
            wjsA = get_wjs(localA_gt, globalA_gt)
            wjsA = tf.reshape(wjsA, [batch_size, max_len, n_joints, 3])

            wjsB = get_wjs(self.localB_rt, self.globalB_rt)
            wjsB = tf.reshape(wjsB, [batch_size, max_len, n_joints, 3])

            ''' Real data '''
            with tf.variable_scope("DIS_motion", reuse=False):

                inpxyz = tf.reduce_mean(wjsA, axis=2)
                motion_real = tf.divide(inpxyz, self.inp_height[:, :, None])

                self.MD_logits_real = self.motion_discriminator(motion_real)
                self.Motion_score_real = tf.nn.sigmoid(self.MD_logits_real)

            ''' Fake data '''
            with tf.variable_scope("DIS_motion", reuse=True):

                tgtxyz = tf.reduce_mean(wjsB, axis=2)
                motion_fake = tf.divide(tgtxyz, self.tgt_height[:, :, None])

                self.MD_logits_fake = self.motion_discriminator(motion_fake)
                self.Motion_score_fake = tf.nn.sigmoid(self.MD_logits_fake)

            self.L_Motion_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.MD_logits_real, labels=tf.ones_like(self.MD_logits_real)))

            self.L_Motion_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.MD_logits_fake, labels=tf.zeros_like(self.MD_logits_fake)))

            self.Disc_motion_loss = self.L_Motion_real + self.L_Motion_fake

            # ========================================================================== #
            # ========================= Total Adversarial Loss ========================= #
            # ========================================================================== #
            ''' Total discriminator loss '''
            self.Disc_total_loss = self.Disc_motion_loss
            self.D_fake_score = self.Motion_score_fake

            ''' fool the motion discriminator'''
            self.Gen_motion_loss = tf.reduce_sum(
                tf.multiply((1 - self.aeReg),
                            tf.nn.sigmoid_cross_entropy_with_logits(
                                logits=self.MD_logits_fake,
                                labels=tf.ones_like(self.MD_logits_fake))))

            ''' Total adversarial loss '''
            self.Gen_total_loss = self.Gen_motion_loss
            self.Gen_total_loss = tf.divide(self.Gen_total_loss,
                                            tf.maximum(tf.reduce_sum(1 - self.aeReg), 1))

            # ========================================================================== #
            # ========================== Perceptual Pose Loss ========================== #
            # ========================================================================== #
            self.L_shape = tf.reduce_sum(
                tf.square(tf.multiply((1 - self.aeReg[:, :, None]) * self.mask[:, :, None],
                                      tf.subtract(self.posefeatureA, self.posefeatureB)))
            )
            self.L_shape = tf.divide(self.L_shape, tf.reduce_sum(self.mask))

            # ========================================================================== #
            # ========================= Reconstruction loss ============================ #
            # ========================================================================== #
            attention_list = [7, 8, 11, 12, 15, 16, 19, 20]  # L/R knee, foot, arm, forearm  (1.95)
            attW = np.ones(22)
            attW[attention_list] = 2

            """ Reconstruction Loss """
            joints_err = tf.reduce_sum(
                tf.square(tf.multiply(self.mask[:, :, None, None], tf.subtract(self.localA_ik, localA_gt))),
                axis=[0, 1, 3]
            )

            self.IK_loss = tf.reduce_sum(attW * joints_err)
            self.IK_loss = tf.divide(self.IK_loss, tf.reduce_sum(self.mask))

            """ For training stability, we chose the same character to the input with p = 0.5 """
            ae_joints_err = tf.reduce_sum(
                tf.square(tf.multiply(self.aeReg[:, :, None, None] * self.mask[:, :, None, None],
                                      tf.subtract(self.localB_rt, localB_gt))),
                axis=[0, 1, 3]
            )
            self.local_ae_loss = tf.reduce_sum(attW * ae_joints_err)
            self.local_ae_loss = tf.divide(self.local_ae_loss,
                                           tf.maximum(tf.reduce_sum(self.aeReg * self.mask), 1))

            self.global_ae_loss = tf.reduce_sum(
                tf.square(tf.multiply(self.aeReg[:, :, None] * self.mask[:, :, None],
                                      tf.subtract(self.normalized_vin, self.normalized_vout)))
            )

            self.global_ae_loss = tf.divide(self.global_ae_loss,
                                            tf.maximum(tf.reduce_sum(self.aeReg * self.mask), 1))

            # ========================================================================== #
            # ====================== Rotation Constraint Loss ========================== #
            # ========================================================================== #
            rads = self.alpha / 180.0
            twistA_loss = tf.reduce_mean(
                tf.square(
                    tf.maximum(0.0,
                               tf.abs(self.euler_y(self.quatA_ik, euler_ord)) - rads * np.pi)))
            twistB_loss = tf.reduce_mean(
                tf.square(
                    tf.maximum(0.0,
                               tf.abs(self.euler_y(self.quatB_rt, euler_ord)) - rads * np.pi)))
            self.twist_loss = twistA_loss + twistB_loss

            # ========================================================================== #
            # ========================= Total Loss Functions =========================== #
            # ========================================================================== #
            ''' Overall base loss '''
            self.base_loss = self.IK_loss \
                             + self.local_ae_loss + self.global_ae_loss \
                             + (self.gamma * self.twist_loss)

            ''' Total loss '''
            self.total_loss = (self.balancing * self.Gen_total_loss) + (20 * self.L_shape) + self.base_loss

            # ========================================================================== #
            # =============================== Optimizer ================================ #
            # ========================================================================== #
            self.allvars = tf.trainable_variables()
            self.gvars = [v for v in self.allvars if "DIS" not in v.name]
            self.dvars = [v for v in self.allvars if "DIS" in v.name]

            if optim_name == "rmsprop":
                goptimizer = tf.train.RMSPropOptimizer(self.learning_rate, name="goptimizer")
                doptimizer = tf.train.RMSPropOptimizer(self.learning_rate, name="doptimizer")
            elif optim_name == "adam":
                goptimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, name="goptimizer")
                doptimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, name="doptimizer")
            else:
                raise Exception("Unknown optimizer!")

            ggradients, gg = zip(*goptimizer.compute_gradients(self.total_loss, var_list=self.gvars))
            ggradients, _ = tf.clip_by_global_norm(ggradients, 25)
            self.goptim = goptimizer.apply_gradients(zip(ggradients, gg))

            ''' added after adding discriminator '''
            dgradients, dg = zip(*doptimizer.compute_gradients(self.Disc_total_loss, var_list=self.dvars))
            dgradients, _ = tf.clip_by_global_norm(dgradients, 25)
            self.doptim = doptimizer.apply_gradients(zip(dgradients, dg))

            # ========================================================================== #
            # =============================== Debugging ================================ #
            # ========================================================================== #
            IK_loss_sum = tf.summary.scalar("losses/IK_loss", self.IK_loss)
            aereg_local_loss_sum = tf.summary.scalar("losses/aereg_local_loss", self.local_ae_loss)
            aereg_global_loss_sum = tf.summary.scalar("losses/aereg_global_loss", self.global_ae_loss)
            twist_loss_sum = tf.summary.scalar("losses/twist_loss", (self.gamma * self.twist_loss))

            Lmotionreal_sum = tf.summary.scalar("losses/motion_real", self.L_Motion_real)
            Lmotionfake_sum = tf.summary.scalar("losses/motion_fake", self.L_Motion_fake)
            Lgen_motion_sum = tf.summary.scalar("losses/motion_gen", self.Gen_motion_loss)

            Lshape_sum = tf.summary.scalar("losses/shape_loss", self.L_shape)

            self.summary = tf.summary.merge(
                [IK_loss_sum,
                 aereg_local_loss_sum,
                 aereg_global_loss_sum,
                 twist_loss_sum,
                 Lmotionreal_sum,
                 Lmotionfake_sum,
                 Lgen_motion_sum,
                 Lshape_sum])

            self.writer = tf.summary.FileWriter(logs_dir, tf.get_default_graph())

            num_param = 0
            for var in self.gvars:
                num_param += int(np.prod(var.get_shape()))
            print "Number of G parameters: " + str(num_param)

            ''' added after adding discriminator '''
            num_param = 0
            for var in self.dvars:
                num_param += int(np.prod(var.get_shape()))
            print "NUMBER OF D PARAMETERS: " + str(num_param)

        self.saver = tf.train.Saver()

    def train(self, sess, seqA, skelA, seqB, skelB, mask, aeReg, inp_height, tgt_height, step):
        feed_dict = dict()

        feed_dict[self.seqA] = seqA
        feed_dict[self.skelA] = skelA
        feed_dict[self.seqB] = seqB
        feed_dict[self.skelB] = skelB
        feed_dict[self.mask] = mask
        feed_dict[self.aeReg] = aeReg

        feed_dict[self.inp_height] = inp_height
        feed_dict[self.tgt_height] = tgt_height

        if self.fake_score > self.margin:
            print("update D")
            feed_dict[self.kp_] = 0.7
            cur_score = self.D_fake_score.eval(feed_dict=feed_dict).mean()
            self.fake_score = 0.99 * self.fake_score + 0.01 * cur_score
            _, summary_str = sess.run([self.doptim, self.summary], feed_dict=feed_dict)

        print("update G")
        feed_dict[self.kp_] = 1.0
        cur_score = self.D_fake_score.eval(feed_dict=feed_dict).mean()
        self.fake_score = 0.99 * self.fake_score + 0.01 * cur_score
        _, summary_str = sess.run([self.goptim, self.summary], feed_dict=feed_dict)

        self.writer.add_summary(summary_str, step)

        mf, mr, mg, shape, base = sess.run(
            [self.L_Motion_fake, self.L_Motion_real, self.Gen_motion_loss,
             self.L_shape, self.base_loss],
            feed_dict=feed_dict
        )

        return mf, mr, mg, shape, base

    def output_debug(self, sess, seqA, skelB, mask, inp_height, tgt_height):
        feed_dict = dict()
        feed_dict[self.seqA] = seqA
        feed_dict[self.skelB] = skelB
        feed_dict[self.mask] = mask

        feed_dict[self.inp_height] = inp_height
        feed_dict[self.tgt_height] = tgt_height

        oursL = self.localB_rt.eval(feed_dict=feed_dict)
        oursG = self.globalB_rt.eval(feed_dict=feed_dict)
        quatsB_n = self.quatB_rt.eval(feed_dict=feed_dict)
        delta = self.deltaQ.eval(feed_dict=feed_dict)
        quatsA_n = self.quatA_ik.eval(feed_dict=feed_dict)

        return oursL, oursG, quatsB_n, quatsA_n, delta

    def load(self, sess, checkpoint_dir, model_name=None):
        print("[*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            if model_name is None:
                model_name = ckpt_name
            self.saver.restore(sess, os.path.join(checkpoint_dir, model_name))
            print("     Loaded model: " + str(model_name))
            return True, model_name
        else:
            return False, None

    def save(self, sess, checkpoint_dir, step):
        model_name = "JMR.model"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def normalized(self, angles):
        lengths = tf.sqrt(tf.reduce_sum(tf.square(angles), axis=-1))
        normalized_angle = angles / lengths[..., None]
        return normalized_angle

    def euler_y(self, angles, order="yzx"):
        q = self.normalized(angles)
        q0 = q[..., 0]
        q1 = q[..., 1]
        q2 = q[..., 2]
        q3 = q[..., 3]

        if order == "xyz":
            ex = atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
            ey = asin(tf.clip_by_value(2 * (q0 * q2 - q3 * q1), -1, 1))
            ez = atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
            return tf.stack(values=[ex, ez], axis=-1)[:, :, 1:]
        elif order == "yzx":
            ex = atan2(2 * (q1 * q0 - q2 * q3), -q1 * q1 + q2 * q2 - q3 * q3 + q0 * q0)
            ey = atan2(2 * (q2 * q0 - q1 * q3), q1 * q1 - q2 * q2 - q3 * q3 + q0 * q0)
            ez = asin(tf.clip_by_value(2 * (q1 * q2 + q3 * q0), -1, 1))
            return ey[:, :, 1:]
        else:
            raise Exception("Unknown Euler order!")

    def euler_rot(self, angles):
        q = self.normalized(angles)
        q0 = q[..., 0]
        q1 = q[..., 1]
        q2 = q[..., 2]
        q3 = q[..., 3]

        ex = atan2(2 * (q1 * q0 - q2 * q3), -q1 * q1 + q2 * q2 - q3 * q3 + q0 * q0)
        ey = atan2(2 * (q2 * q0 - q1 * q3), q1 * q1 - q2 * q2 - q3 * q3 + q0 * q0)
        ez = asin(tf.clip_by_value(2 * (q1 * q2 + q3 * q0), -1, 1))

        rotx = ex[..., :]
        roty = ey[..., :]
        rotz = ez[..., :]

        rot = tf.stack([rotx, roty, rotz], axis=-1)
        return rot

    def shape_discriminator(self, input_):
        if self.norm_type == "batch_norm":
            from ops import batch_norm as norm
        elif self.norm_type == "instance_norm":
            from ops import instance_norm as norm
        else:
            raise Exception("Unknown normalization layer!!!")

        if not self.norm_type == "instance_norm":
            input_ = tf.nn.dropout(input_, self.kp_)

        h0 = lrelu(conv1d(input_, 32, k_w=4, name="h0"))
        h1 = lrelu(norm(conv1d(h0, 64, k_w=4, name="h1"), "bn1"))
        h2 = lrelu(norm(conv1d(h1, 128, k_w=4, name="h2"), "bn2"))
        h3 = lrelu(norm(conv1d(h2, 256, k_w=4, name="h3"), "bn3"))
        logits = conv1d(h3, 1, k_w=4, name="logits", padding="VALID")

        return tf.reshape(logits, [self.batch_size, 1])

    def motion_discriminator(self, input_):
        if self.norm_type == "batch_norm":
            from ops import batch_norm as norm
        elif self.norm_type == "instance_norm":
            from ops import instance_norm as norm
        else:
            raise Exception("Unknown normalization layer!!!")

        if not self.norm_type == "instance_norm":
            input_ = tf.nn.dropout(input_, self.kp_)

        h0 = lrelu(conv1d(input_, 16, k_w=4, name="h0"))
        h1 = lrelu(norm(conv1d(h0, 32, k_w=4, name="h1"), "bn1"))
        h2 = lrelu(norm(conv1d(h1, 64, k_w=4, name="h2"), "bn2"))
        h3 = lrelu(norm(conv1d(h2, 64, k_w=4, name="h3"), "bn3"))
        logits = conv1d(h3, 1, k_w=4, name="logits", padding="VALID")

        return tf.reshape(logits, [self.batch_size, 1])




