import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import colorsys
#from apifish.plot.utils import get_minmax_values

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from matplotlib.figure import Figure as figure

import matplotlib.colors as mcolors
from napari.utils.colormaps import Colormap

class Plots:
    
    def plot_intensities(self, bulk_fish_int: list, fig: figure, axes: np.ndarray, lign: int, col: int,
                                  leg_x: list, leg_y: list, title: str, color='red', color_mean= '#FF9D23', bins= 50, shape=None, max_x=None, num_decimals=0):
        """
            figure and axis are received as arguments
            Spot size:  positive float.
        """
        if shape is None:
            max_lign = 2
            max_col = 2
        else:
            max_lign = shape[0]-1
            max_col = shape[1]-1
        
        mean = np.mean(bulk_fish_int)
        axes[lign, col].hist(bulk_fish_int, bins = bins, color=color)
        n, bins_t = np.histogram(bulk_fish_int, bins=bins)        
        
        if lign == 0 and col ==max_col:
            axes[lign, col].axvline(mean, color=color_mean, linestyle='-', linewidth=2, label= 'mean')
            axes[lign, col].legend(loc='upper right', fontsize=6)
        else:
            axes[lign, col].axvline(mean, color=color_mean, linestyle='-', linewidth=2)
        
        if max_x is not None:
            axes[lign, col].set_xlim([0, max_x])

        if col==0:
            if isinstance(leg_y, list):
                axes[lign, col].set_ylabel(leg_y[col])
            elif isinstance(leg_y, str):
                axes[lign, col].set_ylabel(leg_y)
        #if lign==max_lign:
        if isinstance(leg_x, list):
            axes[lign, col].set_xlabel(leg_x[lign])
        elif isinstance(leg_x, str):
            axes[lign, col].set_xlabel(leg_x)

        if col==0 and lign==0:   
            axes[lign, col].set_title(title)
                   
        mean_r = np.round(mean, decimals=num_decimals)

        xticks = axes[lign, col].get_xticks()
        xticklabels  = axes[lign, col].get_xticklabels()
        xtickslabels = [label.get_text() for label in xticklabels]

        xticks = np.append(xticks, mean_r)
        xtickslabels.append(str(mean_r))

        axes[lign, col].set_xticks(xticks)
        axes[lign, col].set_xticklabels(xtickslabels)
        xticklabels = axes[lign, col].get_xticklabels()

        xticklabels[-1].set_color(color_mean) 
        if max_x is None:
            axes[lign, col].set_xlim(0, bins_t[-1])
        
        axes[lign, col].tick_params(axis='x', rotation=45)
    
    
    def violin_plot_intensities(self, ints, figsize=(14,3), exp_name=None, rotation=85, names_short= None, color='k', ymin = 0, ymax=None):
        fig, ax = plt.subplots(figsize=figsize)
        violinplot = ax.violinplot(ints, showmedians=True, showextrema=False)

        for i, group in enumerate(ints):
            x_coords = np.ones_like(group) * (i + 1) + np.random.normal(0, 0.01, size=len(group))
            for ind in range(len(x_coords)):
                ax.plot(x_coords[ind], group[ind], color=color, marker='.', markersize = 1)
            
        if ymax is not None:
            ax.set_ylim([ymin, ymax])    
            
        ax.set_xlabel("Conditions")
        ax.set_ylabel(f"{exp_name}")
        
        if names_short is not None:
            ax.set_xticks(np.arange(1, len(ints)+1 ))
            ax.set_xticklabels(names_short, rotation=rotation)
        
        return fig          
    
    
    def plot_hist_num_dots_nuclei(self, count_spots: np.ndarray, fig: figure, axes: np.ndarray, lign: int, col: int,
                                  leg_x: list, leg_y: list, title: str, color='red', color_mean= '#FF9D23', bins= 50,
                                  struct = 'nuclei', shape = None):
        """
            figure and axis are received as arguments
            Spot size:  positive float.
        """
        mean = np.mean(count_spots)
        axes[lign, col].hist(count_spots, bins = bins, color=color)
        n, bins_t = np.histogram(count_spots, bins=bins)        
        
        if shape is None:
            max_lign = 2
            max_col = 2
        else:
            max_lign = shape[0]-1
            max_col = shape[1]-1
        
        if lign == 0 and col ==max_col:
            axes[lign, col].axvline(mean, color=color_mean, linestyle='-', linewidth=2, label= 'mean spot/'+struct)
            axes[lign, col].legend(loc='upper right', fontsize=6)
        else:
            axes[lign, col].axvline(mean, color=color_mean, linestyle='-', linewidth=2)

        if col==0:
            if isinstance(leg_y, list):
                axes[lign, col].set_ylabel(leg_y[lign])
            elif isinstance(leg_y, str):
                axes[lign, col].set_ylabel(leg_y)
        if isinstance(leg_x, list):
            axes[lign, col].set_xlabel(leg_x[col])
        elif isinstance(leg_x, str):
            axes[lign, col].set_xlabel(leg_x)
                
        if col==0 and lign==0:   
            axes[lign, col].set_title(title + '   # spots / '+ struct )
                    
        mean_r = np.floor(mean * 10) / 10
        xticks = axes[lign, col].get_xticks()
        xticklabels  = axes[lign, col].get_xticklabels()
        xtickslabels = [label.get_text() for label in xticklabels]

        xticks = np.append(xticks, mean_r)
        xtickslabels.append(str(mean_r))

        axes[lign, col].set_xticks(xticks)
        axes[lign, col].set_xticklabels(xtickslabels, rotation=60)
        xticklabels = axes[lign, col].get_xticklabels()

        xticklabels[-1].set_color(color_mean) 
        axes[lign, col].set_xlim([0, bins_t[-1]])
        
    def plot_subcellular_localization(self, g_spots_in_masks: np.ndarray, fig: figure,
                                      axes: np.ndarray, lign: int, col: int, leg_x: list,
                                      leg_y: list, title: str,  color='red', struct = '   number of rna outside or inside the cells',
                                      shape=None):
        """
        Count the number of rnas inside vs outside cells.
        """    
        if shape is None:
            max_lign = 2
            max_col = 2
        else:
            max_lign = shape[0]-1
            max_col = shape[1]-1
               
        bins = np.array([-0.3, 0.3, 0.7, 1.3]) 
        axes[lign, col].hist(g_spots_in_masks , bins=bins, color=color);
        axes[lign, col].set_xticks([0, 1])
        
        if struct == '   number of rna outside or inside the cells':
            axes[lign, col].set_xticklabels(['Out', 'In']) 
        else:
            axes[lign, col].set_xticklabels(['Cyto', 'Nucleus']) 
        
        if col==0:
            if isinstance(leg_y, list):
                axes[lign, col].set_ylabel(leg_y[lign])
            elif isinstance(leg_y, str):
                axes[lign, col].set_ylabel(leg_y)
        #if lign==max_lign:
        if isinstance(leg_x, list):
            axes[lign, col].set_xlabel(leg_x[col])
        else:
            axes[lign, col].set_xlabel(leg_x)
                
        if col==0 and lign==0:   
            axes[lign, col].set_title(title + struct )    
    
