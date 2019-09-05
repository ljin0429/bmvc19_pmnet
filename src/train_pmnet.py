import os
import sys
sys.path.insert(0, '../outside-code')
import matplotlib
matplotlib.use("Agg")
import time
import numpy as np
import tensorflow as tf
from model_pmnet import pmnet_model
from os import listdir, makedirs
from os.path import exists
from utils import get_minibatches_idx


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def get_height(joints):
    return (np.sqrt(((joints[5, :] - joints[4, :]) ** 2).sum(axis=-1)) +
            np.sqrt(((joints[4, :] - joints[3, :]) ** 2).sum(axis=-1)) +
            np.sqrt(((joints[3, :] - joints[2, :]) ** 2).sum(axis=-1)) +
            np.sqrt(((joints[2, :] - joints[1, :]) ** 2).sum(axis=-1)) +
            np.sqrt(((joints[1, :] - joints[0, :]) ** 2).sum(axis=-1)) +
            np.sqrt(((joints[6, :] - joints[7, :]) ** 2).sum(axis=-1)) +
            np.sqrt(((joints[7, :] - joints[8, :]) ** 2).sum(axis=-1)) +
            np.sqrt(((joints[8, :] - joints[9, :]) ** 2).sum(axis=-1))
            )


def get_height_from_skel(skel):
    diffs = np.sqrt((skel**2).sum(axis=-1))
    height = diffs[1:6].sum() + diffs[7:10].sum()
    return height


def main(gpu, mem_frac, batch_size, alpha, gamma, omega, euler_ord, max_len, optimizer, keep_prob, learning_rate,
         margin, norm_type, balancing):
    # ========================================================================== #
    # =============================== Parameters =============================== #
    # ========================================================================== #
    params = {'gpu': gpu,
              'mem_frac': mem_frac,
              'batch_size': batch_size,
              'alpha': alpha,   # threshold for euler angle
              'gamma': gamma,   # weight factor for twist loss
              'omega': omega,   # weight factor for smooth loss
              'euler_ord': euler_ord,
              'max_len': max_len,
              'optimizer': optimizer,
              'keep_prob': keep_prob,
              'learning_rate': learning_rate,
              'margin': margin,
              'norm_type': norm_type,
              'balancing': balancing}

    prefix = "pmnet"
    for k, v in params.items():
        if (k != 'gpu' and k != 'mem_frac' and k != 'euler_ord'):
            prefix += "_" + k + "=" + str(v)
    # ========================================================================== #
    # =============================== Load Data ================================ #
    # ========================================================================== #
    data_path = "../datasets/train/"
    stats_path = "../data/"

    # Mixamo joint configuration
    parents = np.array([-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 0, 10, 11, 12, 3, 14, 15, 16, 3, 18, 19, 20])

    all_local = []
    all_global = []
    all_skel = []
    all_names = []
    t_skel = []

    folders = [
        f for f in listdir(data_path)
        if not f.startswith(".") and not f.endswith("py") and not f.endswith(".npz")
    ]
    for folder_name in folders:
        files = [
            f for f in listdir(data_path + folder_name)
            if not f.startswith(".") and f.endswith("_seq.npy")
        ]
        for cfile in files:
            file_name = cfile[:-8]
            # Real joint positions
            positions = np.load(data_path + folder_name + "/" + file_name + "_skel.npy")

            # After processed (Maybe, last 4 elements are dummy values)
            sequence = np.load(data_path + folder_name + "/" + file_name + "_seq.npy")

            # Processed global positions (#frames, 4)
            offset = sequence[:, -8:-4]

            # Processed local positions (#frames, #joints, 3)
            sequence = np.reshape(sequence[:, :-8], [sequence.shape[0], -1, 3])
            positions[:, 0, :] = sequence[:, 0, :]    # root joint

            all_local.append(sequence)
            all_global.append(offset)
            all_skel.append(positions)
            all_names.append(folder_name)

    # Joint positions before processed
    train_skel = all_skel

    # After processed, relative position
    train_local = all_local
    train_global = all_global

    # T-pose (real position)
    for tt in train_skel:
        t_skel.append(tt[0:1])

    # Total training samples
    all_frames = np.concatenate(train_local)
    ntotal_samples = all_frames.shape[0]
    ntotal_sequences = len(train_local)
    print("Number of sequences: " + str(ntotal_sequences))

    # ========================================================================== #
    # ============================= Data Normalize ============================= #
    # ========================================================================== #
    # Calculate total mean and std
    allframes_n_skel = np.concatenate(train_local + t_skel)
    local_mean = allframes_n_skel.mean(axis=0)[None, :]
    global_mean = np.concatenate(train_global).mean(axis=0)[None, :]
    local_std = allframes_n_skel.std(axis=0)[None, :]
    global_std = np.concatenate(train_global).std(axis=0)[None, :]

    # Save the data stats
    np.save(stats_path + "mixamo_local_motion_mean.npy", local_mean)
    np.save(stats_path + "mixamo_local_motion_std.npy", local_std)
    np.save(stats_path + "mixamo_global_motion_mean.npy", global_mean)
    np.save(stats_path + "mixamo_global_motion_std.npy", global_std)

    # Normalize the data (whitening)
    n_joints = all_local[0].shape[-2]
    local_std[local_std == 0] = 1

    for i in xrange(len(train_local)):
        train_local[i] = (train_local[i] - local_mean) / local_std
        train_global[i] = (train_global[i] - global_mean) / global_std
        train_skel[i] = (train_skel[i] - local_mean) / local_std

    # ========================================================================== #
    # =============================== Load Model =============================== #
    # ========================================================================== #
    models_dir = "../data/models/" + prefix
    logs_dir = "../data/logs/" + prefix

    if not exists(models_dir):
        makedirs(models_dir)

    if not exists(logs_dir):
        makedirs(logs_dir)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_frac)

    with tf.device("/gpu:%d" % gpu):
        net = pmnet_model(
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
            optimizer,
            local_mean,
            local_std,
            global_mean,
            global_std,
            logs_dir,
            margin,
            norm_type,
            balancing
        )

    # ========================================================================== #
    # ================================ Training ================================ #
    # ========================================================================== #
    with tf.Session(
            config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False,
                gpu_options=gpu_options)) as sess:

        sess.run(tf.global_variables_initializer())

        loaded, model_name = net.load(sess, models_dir)
        if loaded:
            print("[*] Load SUCCESSFUL")
            iteration = int(model_name.split("-")[-1])
        else:
            print("[!] Starting from scratch ...")
            iteration = 0

        net.saver = tf.train.Saver(max_to_keep=10)

        max_iter = 15000

        while iteration < max_iter:
            mini_batches = get_minibatches_idx(len(train_local), batch_size, shuffle=True)

            for _, batch_idxs in mini_batches:
                start_time = time.time()
                if len(batch_idxs) == batch_size:

                    steps = np.repeat(max_len, batch_size)

                    localA_batch = []
                    globalA_batch = []
                    skelA_batch = []
                    localB_batch = []
                    globalB_batch = []
                    skelB_batch = []
                    mask_batch = np.zeros((batch_size, max_len), dtype="float32")
                    aeReg_batch = np.zeros((batch_size, 1), dtype="float32")

                    inp_height_batch = np.zeros((batch_size, 1), dtype="float32")
                    tgt_height_batch = np.zeros((batch_size, 1), dtype="float32")

                    # Make minibatch
                    for bb in xrange(batch_size):
                        low = 0
                        high = train_local[batch_idxs[bb]].shape[0] - max_len
                        if low >= high:
                            stidx = 0
                        else:
                            stidx = np.random.randint(low=low, high=high)

                        clocalA = train_local[batch_idxs[bb]][stidx:(stidx + max_len)]
                        mask_batch[bb, :np.min([max_len, clocalA.shape[0]])] = 1.0

                        if clocalA.shape[0] < max_len:
                            clocalA = np.concatenate((clocalA,
                                                      np.zeros((max_len - clocalA.shape[0], n_joints, 3))))

                        cglobalA = train_global[batch_idxs[bb]][stidx:(stidx + max_len)]
                        if cglobalA.shape[0] < max_len:
                            cglobalA = np.concatenate((cglobalA,
                                                      np.zeros((max_len - cglobalA.shape[0], n_joints, 3))))

                        cskelA = train_skel[batch_idxs[bb]][stidx:(stidx + max_len)]
                        if cskelA.shape[0] < max_len:
                            cskelA = np.concatenate((cskelA,
                                                     np.zeros((max_len - cskelA.shape[0], n_joints, 3))))

                        rnd_idx = np.random.randint(len(train_local))

                        cskelB = train_skel[rnd_idx][0:max_len]
                        if cskelB.shape[0] < max_len:
                            cskelB = np.concatenate((cskelB,
                                                     np.zeros((max_len - cskelB.shape[0], n_joints, 3))))

                        joints_a = cskelA[0].copy()
                        joints_a = joints_a[None]
                        joints_a = (joints_a * local_std) + local_mean
                        height_a = get_height_from_skel(joints_a[0])
                        height_a = height_a / 100

                        joints_b = cskelB[0].copy()
                        joints_b = joints_b[None]
                        joints_b = (joints_b * local_std + local_mean)
                        height_b = get_height_from_skel(joints_b[0])
                        height_b = height_b / 100

                        aeReg_on = np.random.binomial(1, p=0.5)
                        if aeReg_on:
                            cskelB = cskelA.copy()
                            aeReg_batch[bb, 0] = 1

                            inp_height_batch[bb, 0] = height_a
                            tgt_height_batch[bb, 0] = height_a
                        else:
                            aeReg_batch[bb, 0] = 0

                            inp_height_batch[bb, 0] = height_a
                            tgt_height_batch[bb, 0] = height_b

                        localA_batch.append(clocalA)
                        globalA_batch.append(cglobalA)
                        skelA_batch.append(cskelA)
                        localB_batch.append(clocalA)
                        globalB_batch.append(cglobalA)
                        skelB_batch.append(cskelB)

                    localA_batch = np.array(localA_batch).reshape((batch_size, max_len, -1))
                    globalA_batch = np.array(globalA_batch).reshape((batch_size, max_len, -1))
                    seqA_batch = np.concatenate((localA_batch, globalA_batch), axis=-1)
                    skelA_batch = np.array(skelA_batch).reshape((batch_size, max_len, -1))

                    localB_batch = np.array(localB_batch).reshape((batch_size, max_len, -1))
                    globalB_batch = np.array(globalB_batch).reshape((batch_size, max_len, -1))
                    seqB_batch = np.concatenate((localB_batch, globalB_batch), axis=-1)
                    skelB_batch = np.array(skelB_batch).reshape((batch_size, max_len, -1))

                    mid_time = time.time()

                    mf, mr, mg, shape, base = net.train(sess,
                                                        seqA_batch,
                                                        skelA_batch,
                                                        seqB_batch,
                                                        skelB_batch,
                                                        mask_batch,
                                                        aeReg_batch,
                                                        inp_height_batch,
                                                        tgt_height_batch,
                                                        iteration
                                                        )

                    print("step=%d/%d, time=%.2f+%.2f" %
                          (iteration, max_iter, mid_time - start_time, time.time() - mid_time))

                    if np.isnan(mg) or np.isinf(mg):
                        return

                    if iteration >= 1000 and iteration % 5000 == 0:
                        net.save(sess, models_dir, iteration)

                    iteration = iteration + 1

        net.save(sess, models_dir, iteration)


if __name__ == "__main__":
    # Parameters
    gpu = 0
    mem_frac = 1.0
    batch_size = 16
    alpha = 100.0
    gamma = 10.0
    omega = 0.0
    euler_ord = "yzx"
    max_len = 60
    optimizer = "adam"
    keep_prob = 0.8
    learning_rate = 0.0001

    margin = 0.3
    norm_type = "batch_norm"
    balancing = 2

    main(gpu, mem_frac, batch_size, alpha, gamma, omega, euler_ord, max_len, optimizer, keep_prob, learning_rate,
         margin, norm_type, balancing)
