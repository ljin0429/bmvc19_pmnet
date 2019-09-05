import os
import sys
sys.path.insert(0, '../outside-code')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import BVH as BVH
import Animation as Animation
from Quaternions import Quaternions
import numpy as np
from os import listdir, makedirs, path
from model_pmnet import pmnet_model
from utils import load_testdata
from utils import put_in_world_bvh
from utils import get_orient_start


def run_test(gpu, prefix, debug_note, mem_frac):
    is_test = True

    results_dir = "../results/outputs/test/" + debug_note

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # ========================================================================== #
    # =============================== Load Data ================================ #
    # ========================================================================== #
    stats_path = "../data/"

    # Loat test data
    min_steps = 120
    max_steps = 120
    (testlocal, testglobal, testoutseq, testskel, from_names, to_names,
     tgtjoints, tgtanims, inpjoints, inpanims, gtanims) = load_testdata(
        min_steps, max_steps)

    # Mixamo joint configuration
    parents = np.array([-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 0, 10, 11, 12, 3, 14, 15, 16, 3, 18, 19, 20])

    # ========================================================================== #
    # ============================= Data Normalize ============================= #
    # ========================================================================== #
    local_mean = np.load(stats_path + "mixamo_local_motion_mean.npy")
    local_std = np.load(stats_path + "mixamo_local_motion_std.npy")
    global_mean = np.load(stats_path + "mixamo_global_motion_mean.npy")
    global_std = np.load(stats_path + "mixamo_global_motion_std.npy")
    local_std[local_std == 0] = 1

    for i in xrange(len(testlocal)):
        testlocal[i] = (testlocal[i] - local_mean) / local_std
        testglobal[i] = (testglobal[i] - global_mean) / global_std
        testskel[i] = (testskel[i] - local_mean) / local_std

    # ========================================================================== #
    # =============================== Load Model =============================== #
    # ========================================================================== #
    models_dir = "../data/models/" + prefix
    keep_prob = 1.0
    n_joints = testskel[0].shape[-2]

    with tf.device("/gpu:%d" % gpu):
        net = pmnet_model(
            1,  # batch_size
            None,  # alpha
            None,  # gamma
            None,  # omega
            None,  # euler_ord
            n_joints,
            max_steps,
            parents,
            keep_prob,
            None,  # learning_rate
            None,  # optimizer
            local_mean,
            local_std,
            global_mean,
            global_std,
            None,   # logs_dir
            None,   # margin
            None,   # norm_type
            None,   # balancing
            is_train=False
        )

    local_mean = local_mean.reshape((1, 1, -1))
    local_std = local_std.reshape((1, 1, -1))

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_frac)
    with tf.Session(
            config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False,
                gpu_options=gpu_options)) as sess:

        sess.run(tf.global_variables_initializer())
        loaded, model_name = net.load(sess, models_dir)

        if loaded:
            print("[*] Load SUCCESS")
        else:
            print("[!] Load failed...")
            return

        # ========================================================================== #
        # ================================== Test ================================== #
        # ========================================================================== #
        for i in xrange(len(testlocal)):
            print("Testing: " + str(i + 1) + "/" + str(len(testlocal)))
            res_path = results_dir + "/{0:05d}".format(i)

            # Make minibatch
            mask_batch = np.zeros((1, max_steps), dtype="float32")
            inp_height_batch = np.zeros((1, 1), dtype="float32")
            tgt_height_batch = np.zeros((1, 1), dtype="float32")

            localA_batch = testlocal[i][:max_steps].reshape([1, max_steps, -1])
            globalA_batch = testglobal[i][:max_steps].reshape([1, max_steps, -1])
            seqA_batch = np.concatenate((localA_batch, globalA_batch), axis=-1)
            skelB_batch = testskel[i][:max_steps].reshape([1, max_steps, -1])

            step = max_steps
            mask_batch[0, :step] = 1.0

            """ Height ratio """
            # Input sequence (un-normalize)
            inp_skel = seqA_batch[0, 0, :-4].copy() * local_std + local_mean
            inp_skel = inp_skel.reshape([22, 3])

            # Ground truth
            gt = testoutseq[i][None, :max_steps, :].copy()
            out_skel = gt[0, 0, :-4].reshape([22, 3])

            inp_height = get_height(inp_skel) / 100
            out_height = get_height(out_skel) / 100

            inp_height_batch[0, 0] = inp_height
            tgt_height_batch[0, 0] = out_height
            # hratio_batch[0, 0] = out_height / inp_height

            """ BVH load """
            tgtanim, tgtnames, tgtftime = tgtanims[i]
            gtanim, gtnames, gtftime = gtanims[i]
            inpanim, inpnames, inpftime = inpanims[i]

            """ Ours prediction """
            oursL, oursG, quatsB, quatsA, delta = net.output_debug(sess,
                                                                   seqA_batch,
                                                                   skelB_batch,
                                                                   mask_batch,
                                                                   inp_height_batch,
                                                                   tgt_height_batch)

            oursL = oursL.reshape([1, 120, -1])

            """ Un-normalize the output and input sequence """
            # Un-normalize the output
            oursL[:, :step, :] = oursL[:, :step, :] * local_std + local_mean
            oursG[:, :step, :] = oursG[:, :step, :] * global_std + global_mean

            # Input sequence (un-normalize)
            seqA_batch[:, :step, :-4] = seqA_batch[:, :step, :-4] * local_std + local_mean
            seqA_batch[:, :step, -4:] = seqA_batch[:, :step, -4:] * global_std + global_mean

            """ results """
            ours_total = np.concatenate((oursL, oursG), axis=-1)

            """ save """
            np.savez(res_path + "_from=" + from_names[i] + "_to=" + to_names[i] + ".npz",
                     ours_=ours_total,
                     gt_=gt
                     )

            """ VIDEO BVH """
            # offset (skel)
            tjoints = np.reshape(skelB_batch * local_std + local_mean, [max_steps, -1, 3])
            bl_tjoints = tjoints.copy()

            # tmp_gt: (120, 67, 3) i.e. total joint positions
            tmp_gt = Animation.positions_global(
                gtanim)  # Given an animation compute the global joint positions at at every frame
            start_rots = get_orient_start(tmp_gt,
                                          tgtjoints[i][14],     # left shoulder
                                          tgtjoints[i][18],     # right shoulder
                                          tgtjoints[i][6],      # left upleg
                                          tgtjoints[i][10])     # right upleg

            """Exclude angles in exclude_list as they will rotate non-existent children During training."""
            exclude_list = [5, 17, 21, 9, 13]  # head, left hand, right hand, left toe base, right toe base
            canim_joints = []
            cquat_joints = []
            for l in xrange(len(tgtjoints[i])):
                if l not in exclude_list:
                    canim_joints.append(tgtjoints[i][l])
                    cquat_joints.append(l)

            outputB_bvh = ours_total[0].copy()   # ours local and global gt

            """Follow the same motion direction as the input and zero speeds
                            that are zero in the input."""
            outputB_bvh[:, -4:] = outputB_bvh[:, -4:] * (np.sign(seqA_batch[0, :, -4:]) * np.sign(ours_total[0, :, -4:]))
            outputB_bvh[:, -3][np.abs(seqA_batch[0, :, -3]) <= 1e-2] = 0.

            outputB_bvh[:, :3] = gtanim.positions[:1, 0, :].copy()
            wjs, rots = put_in_world_bvh(outputB_bvh.copy(), start_rots)
            tjoints[:, 0, :] = wjs[0, :, 0].copy()

            cpy_bvh = seqA_batch[0].copy()
            cpy_bvh[:, :3] = gtanim.positions[:1, 0, :].copy()
            bl_wjs, _ = put_in_world_bvh(cpy_bvh.copy(), start_rots)
            bl_tjoints[:, 0, :] = bl_wjs[0, :, 0].copy()

            """ Quaternion results """
            cquat = quatsB[0][:, cquat_joints].copy()

            if "Big_Vegas" in from_names[i]:
                from_bvh = from_names[i].replace("Big_Vegas", "Vegas")
            else:
                from_bvh = from_names[i]

            if "Warrok_W_Kurniawan" in to_names[i]:
                to_bvh = to_names[i].replace("Warrok_W_Kurniawan", "Warrok")
            else:
                to_bvh = to_names[i]

            bvh_path = "../results/bvh_files/" + debug_note + "/" + \
                       to_bvh.split("_")[-1]
            if not os.path.exists(bvh_path):
                os.makedirs(bvh_path)

            bvh_path += "/{0:05d}".format(i)

            """ Input bvh file"""
            BVH.save(bvh_path + "_from=" + from_bvh + "_to=" + to_bvh + "_inp.bvh",
                     inpanim, inpnames, inpftime)

            """ GT bvh file"""
            BVH.save(bvh_path + "_from=" + from_bvh + "_to=" + to_bvh + "_gt.bvh",
                     gtanim, gtnames, gtftime)

            """ Copy baseline bvh file """
            tgtanim.positions[:, tgtjoints[i]] = bl_tjoints.copy()
            tgtanim.offsets[tgtjoints[i][1:]] = bl_tjoints[0, 1:]

            BVH.save(bvh_path + "_from=" + from_bvh + "_to=" + to_bvh + "_cpy.bvh",
                     tgtanim, tgtnames, tgtftime)

            """ Ours bvh file """
            tgtanim.positions[:, tgtjoints[i]] = tjoints
            tgtanim.offsets[tgtjoints[i][1:]] = tjoints[0, 1:]
            cquat[:, 0:1, :] = (rots * Quaternions(cquat[:, 0:1, :])).qs
            tgtanim.rotations.qs[:, canim_joints] = cquat

            BVH.save(bvh_path + "_from=" + from_bvh + "_to=" + to_bvh + ".bvh",
                     tgtanim, tgtnames, tgtftime)

        print("[*] Test Done.")


def get_bonelengths(joints):
    # Mixamo config
    parents = np.array([-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 0, 10, 11, 12, 3, 14, 15, 16, 3, 18, 19, 20])

    c_offsets = []
    for j in xrange(parents.shape[0]):
        if parents[j] != -1:
            c_offsets.append(joints[:, :, j, :] - joints[:, :, parents[j], :])
        else:
            c_offsets.append(joints[:, :, j, :])

    offsets = np.stack(c_offsets, axis=2)

    return np.sqrt(((offsets) ** 2).sum(axis=-1))[..., 1:]


def compare_bls(bl1, bl2):
    relbones = np.array([-1, 0, 1, 2, 3, -1, 5, 6, 7, -1, 9, 10, 11, -1, 13, 14, 15, -1, 17, 18, 19])
    bl_diff = np.abs(bl1 - bl2).mean()

    bl1ratios = []
    bl2ratios = []
    for j in xrange(len(relbones)):
        if relbones[j] != -1:
            bl1ratios.append(bl1[j] / bl1[relbones[j]])
            bl2ratios.append(bl2[j] / bl2[relbones[j]])

    blratios_diff = np.abs(np.stack(bl1ratios) - np.stack(bl2ratios)).mean()

    return bl_diff, blratios_diff


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


def put_in_world(states):
    joints = states[:, :-4]
    root_x = states[:, -4]
    root_y = states[:, -3]
    root_z = states[:, -2]
    root_r = states[:, -1]

    joints = joints.reshape(joints.shape[:1] + (-1, 3))

    rotation = Quaternions.id(1)
    offsets = []
    translation = np.array([[0, 0, 0]])

    for i in range(len(joints)):
        joints[i, :, :] = rotation * joints[i]
        joints[i, :, 0] = joints[i, :, 0] + translation[0, 0]
        joints[i, :, 1] = joints[i, :, 1] + translation[0, 1]
        joints[i, :, 2] = joints[i, :, 2] + translation[0, 2]
        rotation = Quaternions.from_angle_axis(-root_r[i], np.array([0, 1, 0])) * rotation
        offsets.append(rotation * np.array([0, 0, 1]))
        translation = translation + rotation * np.array([root_x[i], root_y[i], root_z[i]])

    return joints[None]


def run_evaluate(debug_note):
    output_dir = "../results/outputs/test/" + debug_note + "/"
    files = sorted([f for f in listdir(output_dir) if f.endswith(".npz")])

    res1 = []

    labels = []
    filenames = []

    if not path.exists("../results/quantitative/test/"):
        makedirs("../results/quantitative/test/")

    # ========================================================================== #
    # ================================ Evaluate ================================ #
    for i in xrange(len(files)):
        filenames.append(files[i])
        from_lbl = files[i].split("from=")[1].split("to=")[0]
        to_lbl = files[i].split("to=")[1].split(".npz")[0]
        labels.append(from_lbl + "/" + to_lbl)

        gt = put_in_world(np.load(output_dir + files[i])["gt_"][0])
        gtheight = get_height(gt[0, 0])

        cres1 = put_in_world(np.load(output_dir + files[i])["ours_"][0])

        res1.append(1. / gtheight * ((cres1 - gt) ** 2).sum(axis=-1))

        print(str(i + 1) + " out of " + str(len(files)) + " processed")

    # ========================================================================== #
    # ============================== Save Results ============================== #
    # Save to text file
    f = open("../results/quantitative/test/" + debug_note + ".txt", "w")

    feet = np.array([9, 13])
    f.write("###########################################################\n")
    f.write("## ALL CHARACTERS.\n")
    f.write("## Total\t\t\t" +
            "{0:.2f}".format(np.concatenate(res1).mean()) + "\n\n")

    ''' for each clip '''
    # for i in xrange(len(files)):
    #     f.write("# %d\t\t\t" % i)
    #     f.write("{0:.2f}".format(np.concatenate(res1).mean(axis=1).mean(axis=1)[i]) + "\n")

    f.write("###########################################################\n\n")

    for label in ["new_motion/new_character",
                  "new_motion/known_character",
                  "known_motion/new_character",
                  "known_motion/known_character"]:
        f.write("###########################################################\n")
        f.write("## " + label.upper() + ".\n")
        f.write("###########################################################\n")
        idxs = [i for i, j in enumerate(labels)
                if label.split("/")[0] in j.split("/")[0] and label.split("/")[1] in j.split("/")[1]]
        f.write("## Number of Examples: " +
                "{0:.2f}".format(len(idxs)) + ".\n")
        f.write("## OURS\t\t\t\t\t" +
                "{0:.2f}".format(np.concatenate(res1)[idxs].mean()) +
                "\n")
        f.write("###########################################################\n")

    f.close()
    print("[*] Evaluate Done.")


if __name__ == "__main__":
    # Parameters
    gpu = 0
    prefix = "pmnet_optimizer=adam_learning_rate=0.0001_batch_size=16_norm_type=batch_norm_max_len=60_balancing=2_alpha=100.0_omega=0.0_margin=0.3_keep_prob=0.8_gamma=10.0"
    debug_note = "pmnet"

    mem_frac = 1.0

    run_test(gpu, prefix, debug_note, mem_frac)
    run_evaluate(debug_note)




