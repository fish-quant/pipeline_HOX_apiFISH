import numpy as np
import pandas as pd

import apifish.stack as stack
from apifish.multistack import extract_cell, summarize_extraction_results
from apifish.detection import get_object_radius_pixel, get_spot_volume, get_spot_surface
from apifish.classification import prepare_extracted_data, features_dispersion

class Synthesis:
    def roi_selection_account(self, df_stats_cells: pd.DataFrame, list_cells_to_rem: np.ndarray):
        """Updates the cell stat dataframe with a column keep (booleans)"""
        df_stats_cells_ext = df_stats_cells.copy()
        df_stats_cells_ext["Keep"] = True
        mask_to_remove = df_stats_cells_ext['Cell_ID'].isin(list_cells_to_rem)
        df_stats_cells_ext.loc[mask_to_remove, 'Keep'] = False
        return df_stats_cells_ext
    
    
    def compute_snr_df(self, image: np.ndarray, df: pd.DataFrame, voxel_size, spot_radius) -> pd.DataFrame:
        df_snr_back = df.copy()
        df_snr_back['snr'] = np.nan
        df_snr_back['background'] = np.nan

        # compute radius used to crop spot image once
        radius_pixel = get_object_radius_pixel(
            voxel_size_nm=voxel_size, object_radius_nm=spot_radius, ndim=image.ndim
        )

        def compute_and_assign(row):
            if not row['in_mask']:
                return pd.Series([np.nan, np.nan])
            y, x = int(row['Y']), int(row['X'])
            z = int(row['Z']) if 'Z' in row and image.ndim == 3 else None
            snr, background = self.compute_snr_back_single_spot(
                image, y, x, radius_pixel, z=z
            )
            return pd.Series([snr, background])
        
        df_snr_back[['snr', 'background']] = df_snr_back.apply(compute_and_assign, axis=1)
        
        return df_snr_back
    
    
    
    def compute_snr_back_single_spot(self, image: np.ndarray, y: int, x: int, radius_pixel, z=None):
        if np.isnan(y) or np.isnan(x) or (z is not None and np.isnan(z)):
            return np.nan, np.nan

        ndim = image.ndim
        spot = np.array([z, y, x] if z is not None else [y, x], dtype=np.int64)

        # Clip coordinates to image bounds
        for i in range(ndim):
            spot[i] = np.clip(spot[i], 0, image.shape[i] - 1)

        # Compute signal and background radii
        radius_signal = np.ceil([np.sqrt(ndim) * r for r in radius_pixel]).astype(int)
        radius_background = (2 * radius_signal).astype(int)

        spot_y, spot_x = spot[-2], spot[-1]
        radius_signal_yx = radius_signal[-1]
        radius_background_yx = radius_background[-1]
        edge_background_yx = radius_background_yx - radius_signal_yx

        if ndim == 3:
            spot_z = spot[0]
            max_signal = image[spot_z, spot_y, spot_x]
            spot_background_, _ = get_spot_volume(
                image, spot_z, spot_y, spot_x, radius_background[0], radius_background_yx
            )
            if spot_background_.shape[1:] != (2 * radius_background_yx + 1, 2 * radius_background_yx + 1):
                return np.nan, np.nan

            # Cast to float64 to allow -1 assignment
            spot_background = spot_background_.astype(np.float64)
            spot_background[:, edge_background_yx:-edge_background_yx, edge_background_yx:-edge_background_yx] = -1

        else:
            max_signal = image[spot_y, spot_x]
            spot_background_, _ = get_spot_surface(image, spot_y, spot_x, radius_background_yx)
            if spot_background_.shape != (2 * radius_background_yx + 1, 2 * radius_background_yx + 1):
                return np.nan, np.nan

            spot_background = spot_background_.astype(np.float64)
            spot_background[edge_background_yx:-edge_background_yx, edge_background_yx:-edge_background_yx] = -1

        # Filter valid background pixels
        valid_background = spot_background[spot_background >= 0]
        if valid_background.size == 0:
            return np.nan, np.nan

        mean_background = np.mean(valid_background)
        std_background  = np.std(valid_background)
        snr = (max_signal - mean_background) / std_background if std_background > 0 else np.nan

        return snr, mean_background
    
    
    
    
    def spatial_statistics(self, mask_cell: np.ndarray, mask_nuc: np.ndarray, df_stat_cells: pd.DataFrame, df_spots: pd.DataFrame, image_2D: np.ndarray):
        """Compute spatial statistics (index_polarization, index_dispersion, index_peripheral_distribution) for each cell with spots on it """
        
        df_stat_cells_ext = df_stat_cells.copy()
        
        ip_list  = []
        id_list  = []
        ipd_list = []

        df_spots_temp = df_spots[df_spots["in_mask"]==True][['cell_mask_num','Y','X']]

        for _, row_cell_stat in df_stat_cells_ext.iterrows():
            ip_ind, id_ind, ipd_ind = self.compute_spatial_stats(row_cell_stat, mask_cell, mask_nuc, df_spots_temp, image_2D)
            ip_list.append(ip_ind)
            id_list.append(id_ind)
            ipd_list.append(ipd_ind)

        df_stat_cells_ext['pol_ind']      = ip_list
        df_stat_cells_ext['disp_ind']     = id_list
        df_stat_cells_ext['per_dist_ind'] = ipd_list

        return df_stat_cells_ext
        
    def compute_spatial_stats(self, row_cell_stat, masks_cells: np.ndarray, masks_nucs: np.ndarray, df_spots: pd.DataFrame, image_2D: np.ndarray):
        
        id_cell   = row_cell_stat['Cell_ID']
        count     = row_cell_stat['counts']
        
        if count > 0:
            
            mask_c = np.zeros_like(masks_cells)
            mask_c[masks_cells == id_cell] = 1

            mask_n = np.zeros_like(masks_cells)
            mask_n[masks_nucs == id_cell] = 1
            
            df_mask =  df_spots[df_spots['cell_mask_num'] == id_cell]
            if len(df_mask):
                spots = df_mask[['Y','X']].to_numpy()
                temp = prepare_extracted_data(mask_c, nuc_mask = mask_n,  rna_coord=spots, ndim=2)
        
                (cell_mask, 
                distance_cell,
                distance_cell_normalized,
                centroid_cell,
                distance_centroid_cell,
                nuc_mask,
                cell_mask_out_nuc,
                distance_nuc,
                distance_nuc_normalized,
                centroid_nuc,
                distance_centroid_nuc,
                rna_coord_out_nuc,
                centroid_rna,
                distance_centroid_rna,
                centroid_rna_out_nuc,
                distance_centroid_rna_out_nuc,
                distance_centrosome) = temp
                
                ndim = 2
                index_polarization, index_dispersion, index_peripheral_distribution = features_dispersion(image_2D, spots, centroid_rna,
                                                                                                          cell_mask, centroid_cell,
                                                                                                          centroid_nuc, ndim, check_input=False)
                return index_polarization, index_dispersion, index_peripheral_distribution
            else:
                raise ValueError('There should be spots in this area')
        else:
            return np.nan, np.nan, np.nan    
  

    
    def binary_colocalization(self, df_g1: pd.DataFrame, df_g2: pd.DataFrame, name_gene1: str, name_gene2: str, df_stats: pd.DataFrame):
        """Boolean variable whether two genes are expressed in the same cell"""
        list_cells_coloc = np.array(list(set(np.unique(df_g1[df_g1['in_cell']==1.0]['cell_mask_num'].to_numpy()).tolist())  &
                                    set(np.unique(df_g2[df_g2['in_cell']==1.0]['cell_mask_num'].to_numpy()).tolist())), dtype= int)    
        df = df_stats.copy()
        df['coloc' + '_' + name_gene1 + '_' + name_gene2] = df['Cell_ID'].isin(list_cells_coloc)*1.0
        
        df.loc[df['Keep'] == False, 'coloc' + '_' + name_gene1 + '_' + name_gene2] = np.nan
        
        return df
    
    
    def colocalization_analysis(
        self, gene1_spot: np.ndarray, gene2_spot: np.ndarray, thresh_dist=None
    ):
        """associates together spots that are closer than a threshold.

        Args:
            gene1_spot (np.ndarray): positions of the first gene detected (of dimensions 2 or 3)
            gene2_spot (np.ndarray): positions of the second gene detected.
            thresh_dist (_type_, optional): Defaults to np.sqrt(2).


        Returns:
            gene1_alone_number (int): number of positions in which the gene 1 is alone.
            gene2_alone_number (int): number of positions in which the gene 2 is alone.
            gene_a_b_together_num (int):  number of positions in which both genes colocalize.
            positions_overlap  (np.ndarray): coordinates of overlapping points.
        """
        if thresh_dist is None:
            n = len(gene1_spot[0])
            thresh_dist = np.ceil(np.sqrt(n) * 100) / 100

        dist_mat = np.zeros((len(gene1_spot), len(gene2_spot)))
        for ind_l, coord_g1 in enumerate(gene1_spot):
            for ind_c, coord_g2 in enumerate(gene2_spot):
                dist_mat[ind_l, ind_c] = self.euclidean_dist(coord_g1, coord_g2)

        dist_mat_thresh = dist_mat.copy()
        dist_mat_thresh[dist_mat > thresh_dist] = np.nan

        dist_mat_thresh_c = dist_mat_thresh.copy()
        gene1_spot_c = gene1_spot.copy()
        gene2_spot_c = gene2_spot.copy()

        list_gene1_only = []
        list_gene2_only = []
        list_gene1_gene2 = []

        while len(dist_mat_thresh_c):
            coord_start = gene1_spot_c[0, :]
            line_dist = dist_mat_thresh_c[0]
            ind_2 = self.nanargmin(line_dist)
            if isinstance(ind_2, np.ndarray):  # proxi for np.array([])
                # remove first element from list 1
                list_gene1_only.append(coord_start)
                gene1_spot_c = np.delete(gene1_spot_c, 0, axis=0)
                gene1_spot_c = np.atleast_2d(gene1_spot_c)
                dist_mat_thresh_c = np.delete(dist_mat_thresh_c, 0, axis=0)
                dist_mat_thresh_c = np.atleast_2d(dist_mat_thresh_c)
            else:
                col_dist = dist_mat_thresh_c[:, ind_2]
                ind_1 = self.nanargmin(col_dist)
                if isinstance(ind_1, np.ndarray):
                    # retirer b
                    list_gene2_only.append(gene2_spot_c[ind_2, :])
                    gene2_spot_c = np.delete(gene2_spot_c, ind_2, axis=0)
                    gene2_spot_c = np.atleast_2d(gene2_spot_c)
                    dist_mat_thresh_c = np.delete(dist_mat_thresh_c, ind_2, axis=1)
                    dist_mat_thresh_c = np.atleast_2d(dist_mat_thresh_c)
                else:
                    # retirer b, a'
                    list_gene1_gene2.append(
                        (gene1_spot_c[ind_1, :], gene2_spot_c[ind_2, :])
                    )
                    gene1_spot_c = np.delete(gene1_spot_c, ind_1, axis=0)
                    gene1_spot_c = np.atleast_2d(gene1_spot_c)
                    gene2_spot_c = np.delete(gene2_spot_c, ind_2, axis=0)
                    gene2_spot_c = np.atleast_2d(gene2_spot_c)
                    dist_mat_thresh_c = np.delete(dist_mat_thresh_c, ind_2, axis=1)
                    dist_mat_thresh_c = np.atleast_2d(dist_mat_thresh_c)
                    dist_mat_thresh_c = np.delete(dist_mat_thresh_c, ind_1, axis=0)
                    dist_mat_thresh_c = np.atleast_2d(dist_mat_thresh_c)

        if len(gene2_spot_c):
            for coords in gene2_spot_c:
                list_gene2_only.append(coords)

        return list_gene1_only, list_gene2_only, list_gene1_gene2
    
    def euclidean_dist(self, x: np.ndarray, y: np.ndarray):
        return np.sqrt(np.sum((x - y) ** 2))
    
    
    
    
    def nanargmin(self, arr, axis=None, nan_position="last"):
        """
        Find the indices of the minimum values along an axis ignoring NaNs.
        Similar to np.nanargmin, but handles edge cases and nan_position more robustly.

        Args:
            arr (numpy.ndarray): The input array.
            axis (int, optional): The axis along which to operate. Defaults to None,
                                meaning find the minimum of the flattened array.
            nan_position (str, optional): Where to put NaNs in the sorted indices.
                                        'first': NaNs at the beginning.
                                        'last': NaNs at the end.
                                        Defaults to 'last'.

        Returns:
            numpy.ndarray: An array of indices of the minimum values.  Returns an
                        empty array if all elements are NaN or the input is empty.
                        Returns a scalar if axis is None.

        Raises:
            ValueError: If nan_position is not 'first' or 'last'.
        """

        if nan_position not in ("first", "last"):
            raise ValueError("nan_position must be 'first' or 'last'")

        arr = np.asanyarray(arr)  # handle potential non-numpy inputs

        if arr.size == 0:  # Handle empty array case
            return np.array(
                []
            )  # Return empty array, consistent with np.nanargmin behavior

        mask = np.isnan(arr)

        if np.all(mask):  # Handle all-NaN case
            if axis is None:
                return np.array([])  # Return empty array for consistency
            else:
                if nan_position == "first":
                    return np.zeros(
                        arr.shape[:axis] + arr.shape[axis + 1 :], dtype=int
                    )  # Return array of zeros
                else:  # nan_position == 'last'
                    return np.full(
                        arr.shape[:axis] + arr.shape[axis + 1 :],
                        arr.shape[axis] - 1,
                        dtype=int,
                    )  # Return array of last index values

        if axis is None:
            # Flatten and handle 1D array
            flat_arr = arr.flatten()
            flat_mask = mask.flatten()
            valid_indices = np.where(~flat_mask)[0]  # Indices of non-NaN values
            if valid_indices.size == 0:  # all NaN
                return np.array([])
            min_index_flat = valid_indices[np.argmin(flat_arr[valid_indices])]
            return min_index_flat

        else:
            # Handle multi-dimensional array and specified axis
            valid_indices = np.where(~mask)

            if valid_indices[0].size == 0:  # all NaN along the axis
                if nan_position == "first":
                    return np.zeros(arr.shape[:axis] + arr.shape[axis + 1 :], dtype=int)
                else:  # nan_position == 'last'
                    return np.full(
                        arr.shape[:axis] + arr.shape[axis + 1 :],
                        arr.shape[axis] - 1,
                        dtype=int,
                    )

            min_indices = np.zeros(arr.shape[:axis] + arr.shape[axis + 1 :], dtype=int)

            for idx in np.ndindex(*arr.shape[:axis], *arr.shape[axis + 1 :]):
                sl = list(idx[:axis]) + [slice(None)] + list(idx[axis:])
                arr_slice = arr[tuple(sl)]
                mask_slice = mask[tuple(sl)]
                valid_slice_indices = np.where(~mask_slice)[
                    0
                ]  # Indices of non-NaN values
                if valid_slice_indices.size > 0:
                    min_index_slice = valid_slice_indices[
                        np.argmin(arr_slice[valid_slice_indices])
                    ]
                    min_indices[idx] = min_index_slice

            return min_indices

    
    