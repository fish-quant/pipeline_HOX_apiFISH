import numpy as np


class Masks:
    @staticmethod
    def remove_mask_from_bound(label_mask: np.ndarray):
        "remove labels when they touch the boundaries"
        label_mask_n = np.zeros_like(label_mask)
        labels = np.unique(label_mask)
        lab_to_rem = []
        im_dims = np.shape(label_mask)

        for u in labels:
            if u != 0:
                lab_temp = (label_mask == u) * 1
                y_c, x_c = np.where((label_mask == u) * 1)
                min_y, max_y = np.min(y_c), np.max(y_c)
                min_x, max_x = np.min(x_c), np.max(x_c)

                if (
                    (min_y == 0)
                    or (max_y == im_dims[0] - 1)
                    or (min_x == 0)
                    or (max_x == im_dims[1] - 1)
                ):
                    lab_to_rem.append(u)
                    continue
        if len(lab_to_rem):
            new_curr_ind = 1
            for u in labels[1:]:
                if u not in lab_to_rem:
                    label_mask_n[label_mask == u] = new_curr_ind
                    new_curr_ind += 1

        new_labels = np.unique(label_mask_n)
        return label_mask_n, new_labels

    @staticmethod
    def remove_mask_num(label_mask: np.ndarray, num: int):
        label_mask_out = label_mask.copy()
        label_mask_out[label_mask == num] = 0
        return label_mask_out

    @staticmethod
    def add_index_and_labels(
        label_mask_old: np.ndarray, label_mask_splitted: np.ndarray
    ):
        "When a mask is splitted in parts (usually two), add the new labels to the label mask"
        label_mask_new = label_mask_old.copy()
        labels = np.unique(label_mask_old)
        labs_to_add = np.unique(label_mask_splitted)
        lab_max = np.max(labels)
        for u in labs_to_add:
            if u > 0:
                lab_max = lab_max + 1
                label_mask_new[label_mask_splitted == u] = lab_max
        return label_mask_new

    @staticmethod
    def relabel(label_mask_old: np.ndarray):
        "relabel a mask of labels in which there are missing labels"
        label_mask_new = np.zeros_like(label_mask_old)
        mask_list = np.unique(label_mask_old)
        lab_num = len(mask_list)
        for ind_u, u in enumerate(mask_list):
            label_mask_new[label_mask_old == u] = ind_u
        return label_mask_new
