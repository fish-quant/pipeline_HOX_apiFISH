import numpy as np
import pandas as pd
from typing import Union, Optional
import warnings
from pathlib import Path
import matplotlib.pyplot as plt    

from skimage.segmentation import expand_labels
from skimage import measure
from scipy.ndimage import median_filter
import cellpose.models as models
from cellpose import denoise


import apifish.stack as stack



class Segmentation:
    
    def segment_with_custom_model(
        self, image: np.ndarray, pretrained_model_path: str, gpu=False
    ):
        """
        Input:
        image: np.ndarray, image containing cells.
        pretrained_model_path: string
        return:
        mask: np.ndarray of the same size of the image. Each cell is labeled
            with a diffent integer.
        """
        model = models.CellposeModel(gpu=gpu, pretrained_model=pretrained_model_path)
        masks, flows, styles = model.eval(image, diameter=None)
        return masks

    def deblur_cellpose(self, image: np.ndarray, diameter=30, gpu=False):
        """
        Input:
        image: nuclei DAPI image of dimensions N.M.
        diameter: approximate diameter in pixels of the cells.
        gpu: boolean.

        Return:
        im_dn: denoised image of dimensions N.M.
        """
        dn = denoise.DenoiseModel(model_type="deblur", gpu=gpu)
        im_dn = dn.eval(image, channels=None, diameter=diameter)
        return im_dn[:, :, 0]
    
    
    def find_all_contours(self, masks: np.ndarray):
        """
        Find contours on a labeled mask. Each mask is taken separately
        and its countours are determined.

        Input:
            masks:  np.ndarray. Each mask has a unique label (integer).

        Output:
            list of contours (np.ndarrays).
        """
        contours_method = []
        padded_masks    = np.pad(masks, pad_width=1, mode='constant', constant_values=0)        

        for mask_temp_num in np.unique(masks):
            if mask_temp_num:
                mask_temp = (padded_masks == mask_temp_num) * 1
                contours_method.append(
                    measure.find_contours(mask_temp.astype(np.uint8), 0.5)
                )
        contours_method = [item for sublist in contours_method for item in sublist]
        return contours_method
    
    
    def dilate_labels(self, image: np.ndarray[int], distance: int = None) -> np.ndarray[int]:
        """Expand labels in label image by distance pixels without overlapping.

        Args:
            image (np.ndarray): image of labels (int), each label stands
            for a given cell or nuclei.
            distance (int, optional): Defaults to None. In this case,
            expand the labels as much as possibe in the image.
            When distance is an int, expand the labels by distance (in pixels).

        Returns:
            np.ndarray[int]: lLabeled array, whith enlarged connected regions.
        """
        if distance is None:
            distance = np.shape(image)[0]

        return expand_labels(image, distance=distance)
    
    def remove_labels_from_masks(self, masks: np.ndarray, label_list: Union[np.ndarray, int]) -> np.ndarray:
        """Remove the labels from the masks.

        Args:
            masks (np.nparray): each mask has a unique label (int).
            label_list (np.ndarray): list of labels to remove.

        Returns:
            np.ndarray: masks with the labels removed
        """
        masks_rem = masks.copy()
        if isinstance(label_list, np.ndarray):
            if len(label_list):
                for lab in label_list:
                    masks_rem[masks_rem==lab] = 0
        return masks_rem  
    
    
    def remove_labels_from_fishmask(self, mask_fish: np.ndarray, masks_cells: np.ndarray, lab_to_rem: np.ndarray) -> np.ndarray:
        """Remove the putative cells that do not match the size and intensity criteria from the fishmask. 
        fishmask is the region to consider the gene expression. 

        Args:
            mask_fish (np.ndarray): rough region of fish expression.
            masks_cells (np.ndarray): masks of expanded segmented nuclei (putative cells).
            lab_to_rem (np.ndarray): Established list of cells to remove. Do not consider those points neither in the 
            fish expression region. 

        Returns:
            np.ndarray: cleaned fishmask.
        """
        mask_fish_cleaned = mask_fish.copy()                            
        if len(lab_to_rem):
            for lab in lab_to_rem:
                mask_fish_cleaned[masks_cells==lab] = 0        
        return mask_fish_cleaned  
    
    
    def add_column_in_mask_fish(self, df_points: pd.DataFrame, mask_fish: np.ndarray):
        condition = df_points.apply(lambda row: mask_fish.astype(bool)[int(row['Y']), int(row['X'])], axis=1)
        df_points['in_mask'] = condition.to_numpy() 
        return df_points
    
    
    def extract_intensities_df(self, df_spots: pd.DataFrame, rna: np.ndarray):
        
        df_spots_w_intensity              = df_spots.copy()
        df_spots_w_intensity['intensity'] = np.nan

        if rna.ndim == 3 and 'Z' in list(df_spots_w_intensity.columns):
            df_spots_w_intensity.loc[df_spots_w_intensity['in_mask'], 'intensity'] =  \
                        df_spots_w_intensity[df_spots_w_intensity['in_mask']].apply(lambda row: rna[int(row['Z']), int(row['Y']), int(row['X'])], axis=1)
        elif  rna.ndim == 2 and 'Z' not in list(df_spots_w_intensity.columns):
            df_spots_w_intensity.loc[df_spots_w_intensity['in_mask'], 'intensity'] =  \
                        df_spots_w_intensity[df_spots_w_intensity['in_mask']].apply(lambda row: rna[int(row['Y']), int(row['X'])], axis=1)
            
        return df_spots_w_intensity  
    
    
    
    def determine_spots_width_frac_signal_df(self, im_rna: np.ndarray, df: np.ndarray, h_window: int, size_median_filt: int, frac: float):
        
        df_spots_w_frac_sign               = df.copy()
        df_spots_w_frac_sign['spot_width'] = np.nan
    
        if 'Z' in list(df.columns) and im_rna.ndim == 3:
            df_spots_w_frac_sign.loc[df['in_mask'], 'spot_width'] = df_spots_w_frac_sign[df_spots_w_frac_sign['in_mask']].apply(lambda row: self.determine_spot_width_frac_sign(im_rna,
                                                                                                                                                                               h_window, 
                                                                                                                                                                               size_median_filt,
                                                                                                                                                                               frac,
                                                                                                                                                                               int(row['X']),
                                                                                                                                                                               int(row['Y']),
                                                                                                                                                                               z= int(row['Z'])),
                                                                                                                                                                               axis=1)        
        elif 'Z' not in list(df.columns) and im_rna.ndim == 2:   
            df_spots_w_frac_sign.loc[df['in_mask'], 'spot_width'] = df_spots_w_frac_sign[df_spots_w_frac_sign['in_mask']].apply(lambda row: self.determine_spot_width_frac_sign(im_rna,
                                                                                                                                                                               h_window, 
                                                                                                                                                                               size_median_filt,
                                                                                                                                                                               frac,
                                                                                                                                                                               int(row['X']),
                                                                                                                                                                               int(row['Y']),
                                                                                                                                                                               ),
                                                                                                                                                                               axis=1)
        return df_spots_w_frac_sign
        
        
    def determine_spot_width_frac_sign(self, image: np.ndarray,  h_window: int, size_median_filt: int, frac: float, x: int, y: int, z= None):
         
        if z is None: 
            dim_y, dim_x = np.shape(image) 
            if (y - h_window < 0) or (x - h_window < 0):
                return np.nan
            elif (y + h_window >= dim_y) or (x + h_window >= dim_x):
                return np.nan
            else:   
                signal_h =  image[y-h_window:y+h_window+1,x]
                signal_v =  image[y,x-h_window:x+h_window+1]
                sub_image = image[y-h_window:y+h_window+1, x-h_window:x+h_window+1]
                signal_d1, signal_d2 = self.extract_diagonals(sub_image, h_window, h_window)
                
                w0 = self.det_spot_width_frac_peak(signal_h, size_median_filt= size_median_filt, frac= frac)
                w1 = self.det_spot_width_frac_peak(signal_v, size_median_filt= size_median_filt, frac= frac)
                w2 = self.det_spot_width_frac_peak(signal_d1, size_median_filt= size_median_filt, frac=frac)
                w3 = self.det_spot_width_frac_peak(signal_d2, size_median_filt= size_median_filt, frac=frac)
                # if necessary (for matplotlib) divide the width by two to obtain the radius
                # in napari the size is the diameter = width.
                width = np.median([w0, w1, w2, w3])   
                return width        
        else:
            dim_z, dim_y, dim_x = np.shape(image) 
            if (y - h_window < 0) or (x - h_window < 0):
                return np.nan
            elif (y + h_window >= dim_y) or (x + h_window >= dim_x):
                return np.nan
            else:   
                signal_h =  image[z, y-h_window:y+h_window+1,x]
                signal_v =  image[z, y,x-h_window:x+h_window+1]
                sub_image = image[z, y-h_window:y+h_window+1, x-h_window:x+h_window+1]
                signal_d1, signal_d2 = self.extract_diagonals(sub_image, h_window, h_window)
                
                w0 = self.det_spot_width_frac_peak(signal_h, size_median_filt= size_median_filt, frac= frac)
                w1 = self.det_spot_width_frac_peak(signal_v, size_median_filt= size_median_filt, frac= frac)
                w2 = self.det_spot_width_frac_peak(signal_d1, size_median_filt= size_median_filt, frac=frac)
                w3 = self.det_spot_width_frac_peak(signal_d2, size_median_filt= size_median_filt, frac=frac)
                # if necessary (for matplotlib) divide the width by two to obtain the radius
                # in napari the size is the diameter = width.
                width = np.median([w0, w1, w2, w3])   
                return width     
             
    def extract_diagonals(self, image, x, y):
        # Get the dimensions of the image
        rows, cols = image.shape
        
        # Extract the main diagonal (top-left to bottom-right)
        main_diag = []
        i, j = x, y
        while i >= 0 and j >= 0:
            main_diag.insert(0, image[i, j])
            i -= 1
            j -= 1
        i, j = x + 1, y + 1
        while i < rows and j < cols:
            main_diag.append(image[i, j])
            i += 1
            j += 1
        
        # Extract the anti-diagonal (top-right to bottom-left)
        anti_diag = []
        i, j = x, y
        while i >= 0 and j < cols:
            anti_diag.insert(0, image[i, j])
            i -= 1
            j += 1
        i, j = x + 1, y - 1
        while i < rows and j >= 0:
            anti_diag.append(image[i, j])
            i += 1
            j -= 1
        
        return main_diag, anti_diag        
        
        
    def det_spot_width_frac_peak(self, signal: np.ndarray, size_median_filt=4, frac=0.5):
        """_summary_

        Args:
            signal (np.ndarray): 1D slice vertical, horizontal or diagonal of a spot.
            size_median_filt (int, optional): windows size used to smoothen the signal. Defaults to 4.

        Returns:
            int: width of the bell shaped signal.
        """
        fac_int         = 10
        signal_int      = self.interpolate_signal(signal, n=fac_int)
 
        
        signal_sym      = (signal_int + np.flip(signal_int))/2
        filtered_signal = median_filter(signal_sym, size=size_median_filt*fac_int)
        
        diff_max        = np.max(filtered_signal)- np.min(filtered_signal)
                        
        signal_higher = filtered_signal > frac*diff_max + np.min(filtered_signal)
        ind_max       = np.argmax(signal_higher)
        
        ind = ind_max
        list_ind_higher = []
        while ind < len(signal_higher) and signal_higher[ind]:
            list_ind_higher.append(ind)
            ind = ind+1
        
        ind = ind_max-1
        while ind >=0 and signal_higher[ind]:
            list_ind_higher.append(ind)
            ind = ind-1
            
        width = len(list_ind_higher)/fac_int
        
        return width
    
    def interpolate_signal(self, signal: np.array, n=11):
        """
        Upsample signal by interpolating it.
        """
        arr = []
        for u in range(len(signal) - 1):
            if u < len(signal) - 2:
                arr.append(np.linspace(signal[u], signal[u + 1], n)[:-1])
            else:
                arr.append(np.linspace(signal[u], signal[u + 1], n))
        return np.concatenate(arr)
    
    
    
    def compute_sum_spot_intensity_df(self, df: pd.DataFrame, im_rna: np.ndarray):
        
        df_sum_intensity = df.copy()
        df_sum_intensity['sum_intensity'] = np.nan
        
        if 'Z' in list(df.columns) and im_rna.ndim == 3:            
            df_sum_intensity.loc[df_sum_intensity['in_mask'], 'sum_intensity'] = df_sum_intensity[df_sum_intensity['in_mask']].apply(lambda row: self.summed_intensity_in_circle(im_rna,
                                                                                                                                                        int(row['Y']),
                                                                                                                                                        int(row['X']),
                                                                                                                                                        radius= row['spot_width']/2,
                                                                                                                                                        z= int(row['Z']),
                                                                                                                                                        ),
                                                                                                                                                        axis=1,
                                                                                                                    )    
        else:
            df_sum_intensity.loc[df_sum_intensity['in_mask'], 'sum_intensity'] = df_sum_intensity[df_sum_intensity['in_mask']].apply(lambda row: self.summed_intensity_in_circle(im_rna,
                                                                                                                                                        int(row['Y']),
                                                                                                                                                        int(row['X']),
                                                                                                                                                        radius= row['spot_width']/2,
                                                                                                                                                        ),
                                                                                                                                                        axis=1,
                                                                                                                    )
        
        return df_sum_intensity
    
    
    
    def summed_intensity_in_circle(self, image, center_y, center_x, radius, z=None):
        """
        Finds the pixel mean intensity within a circle in a 2D array (image).

        Args:
            image_shape: Tuple (height, width) representing the image dimensions.
            center_y: Y-coordinate of the circle's center.
            center_x: X-coordinate of the circle's center.
            radius: Radius of the circle.

        Returns:
            Returns np.nan if the inputs are not valid.
        """
        if image.ndim == 2:
            image_shape = np.shape(image)

            if np.isnan(radius):
                return np.nan
            if np.isnan(center_y) or np.isnan(center_x):
                return np.nan
            if not np.isnan(radius) and radius <= 0:
                return np.nan

            height, width = image_shape
            y, x = np.ogrid[:height, :width]
            distance = np.sqrt(
                (x - center_x +.5) ** 2 + (y - center_y +.5) ** 2
            )  # Calculate the distance from each pixel to the center
            mask = distance <= radius
                       
            y_coords, x_coords = np.where(mask)
            if len(y_coords):
                return np.sum(image[y_coords, x_coords])

            else:
                return np.nan
            
        elif image.ndim == 3 and z is not None:
            image_shape = np.shape(image)

            if np.isnan(radius):
                return np.nan
            if np.isnan(center_y) or np.isnan(center_x):
                return np.nan
            if not np.isnan(radius) and radius <= 0:
                return np.nan

            depth, height, width = image_shape
            y, x = np.ogrid[:height, :width]
            distance = np.sqrt(
                (x - center_x + .5) ** 2 + (y - center_y + 0.5) ** 2
            )  # Calculate the distance from each pixel to the center
            mask = distance <=  radius
            y_coords, x_coords = np.where(mask)
            if len(y_coords):
                return np.sum(image[z, y_coords, x_coords])
            else:
                return np.nan   
     
    def count_spots_in_masks_df(self, masks: np.ndarray, df_stat_cells: pd.DataFrame, df_spots: pd.DataFrame):
        """
        
        Args:
            masks (np.ndarray): np array with masks. Each integer (constant inside the mask) corresponds to a single structure. 
            df_stat_cells (pd.DataFrame): collects statistics about the cell.
            df_spots (pd.DataFrame): collect statistics about the spots.
        """
        df_stat_cells_w_counts  = df_stat_cells.copy()
        df_spots_w_mask_num     = df_spots.copy()  
               
        df_spots_w_mask_num['in_cell']  = 0
        df_spots_w_mask_num.loc[df_spots_w_mask_num['in_mask'] == False, 'in_cell'] = np.nan
                       
        df_stat_cells_w_counts['counts']     = np.nan
        df_spots_w_mask_num['cell_mask_num'] = np.nan
                    
        for ind_mask, row_mask in df_stat_cells_w_counts.iterrows():    
            mask_lab  = row_mask['Cell_ID']
            tot_count = 0
            for index, spot_row in df_spots_w_mask_num.iterrows():
                if spot_row['in_mask']:
                    if masks[int(spot_row['Y']), int(spot_row['X'])] == mask_lab:
                        tot_count = tot_count + 1
                        df_spots_w_mask_num.at[index, 'in_cell']       = 1.0    
                        df_spots_w_mask_num.at[index, 'cell_mask_num'] = mask_lab

                               
            df_stat_cells_w_counts.at[ind_mask, 'counts'] = tot_count            
                            
                            
        return df_stat_cells_w_counts, df_spots_w_mask_num
    
    def spots_in_nuclei_df(self, mask_nuc: np.ndarray, df_stat_cells: pd.DataFrame, df_spots: pd.DataFrame):
        
        df_spots_w_nuclei = df_spots.copy()
        df_spots_w_nuclei['in_nuclei'] = 0
        df_spots_w_nuclei.loc[df_spots_w_nuclei['in_mask'] == False, 'in_nuclei'] = np.nan
        df_spots_w_nuclei.loc[df_spots_w_nuclei['in_cell'] == 0, 'in_nuclei']     = np.nan
         
        df_stat_cells_w_nuclei_counts = df_stat_cells.copy()
        df_stat_cells_w_nuclei_counts['count_nuclei'] = 0

        for index, spot_row in df_spots_w_nuclei.iterrows():
            if not pd.isna(spot_row['cell_mask_num']):
                
                mask  = (mask_nuc == spot_row['cell_mask_num'])
                if mask[int(spot_row['Y']), int(spot_row['X'])]:
                    df_spots_w_nuclei.at[index, 'in_nuclei'] = 1

        df_spots_w_nuclei['in_cyto'] = df_spots_w_nuclei['in_cell'] - df_spots_w_nuclei['in_nuclei']

        for ind_mask, mask_row in df_stat_cells_w_nuclei_counts.iterrows():
            if mask_row['counts']:
                df_stat_cells_w_nuclei_counts.at[ind_mask, 'count_nuclei'] =  df_spots_w_nuclei[df_spots_w_nuclei['cell_mask_num'] == mask_row['Cell_ID']]['in_nuclei'].sum()
                
        df_stat_cells_w_nuclei_counts['count_cyto'] = df_stat_cells_w_nuclei_counts['counts'] - df_stat_cells_w_nuclei_counts['count_nuclei']

        return df_stat_cells_w_nuclei_counts, df_spots_w_nuclei
    
    
    # def segment_cellpose_cyto3(self, image: np.ndarray, diameter=50):
    #     """
    #     Input:
    #     image: np.ndarray, image containing cells.
    #     diameter: approximate diameter in pixels of the cells.

    #     return:
    #     mask: np.ndarray of the same size of the image. Each cell is labeled
    #         with a diffent integer.
    #     """
    #     model = models.Cellpose(model_type="cyto3")
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore", FutureWarning)
    #         masks, _, _, _ = model.eval(image, diameter=diameter, channels=None)
    #     return masks

    # def segment_cellpose_other_pretrained_models(
    #     self, image: np.ndarray, gpu=False, diameter=50, model_type="cyto2_cp3"
    # ):
    #     """
    #     Input:
    #     image: np.ndarray, image containing cells.
    #     diameter: approximate diameter in pixels of the cells.
    #     model_type: can be 'bact_fluor_cp3', 'cyto2_cp3'
    #     return:
    #     mask: np.ndarray of the same size of the image. Each cell is labeled
    #         with a diffent integer.
    #     """
    #     model = models.CellposeModel(model_type=model_type)  # insterad of Cellpose
    #     masks, flows, styles = model.eval(image, diameter=diameter, channels=None)
    #     return masks

    
  