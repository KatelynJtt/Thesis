import pandas as pd
import  numpy as np
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import rasterio
import os
import numpy as np
import geopandas as gpd
from rasterio.mask import mask
from matplotlib.cm import ScalarMappable
from rasterio.plot import show
import threading

from flask import send_file


def otsu(data):
    """
    Input: 
        data: vegetation index data.
    
    Output:
        optimal_threshold: the computed threshold using Otsu's method.
    """
    hist, bins = np.histogram(data, bins=100, range=(np.min(data), np.max(data)))
    total_freq = sum(hist)
    probabilities = hist / total_freq
    max_variance = 0
    optimal_threshold = 0

    for t in range(1, len(bins)):
        w0 = np.sum(probabilities[:t])
        w1 = np.sum(probabilities[t:])
        if w0 == 0 or w1 == 0:
            continue
        mean0 = np.sum(probabilities[:t] * bins[:t]) / w0
        mean1 = np.sum(probabilities[t:] * bins[t:-1]) / w1
        variance = w0 * w1 * (mean0 - mean1) ** 2
        if variance > max_variance:
            max_variance = variance
            optimal_threshold = bins[t]

    return optimal_threshold


def plot_VI_hist(data, title, plot_index, output_path):
    """
    Input:
        data: vegetation index data.
        title: title of the plot.
        plot_index: index of the plot for identification.
        output_path: path to save the generated plot image.
    """
    optimal_threshold = otsu(data)
    hist, bins = np.histogram(data, bins=100, range=(np.min(data), np.max(data)))

    # Create the plot
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / len(hist)) for i in range(len(hist))]
    if title in ['ExR', 'CIVE']:
        colors = colors[::-1]

    ax.bar(bins[:-1], hist, width=np.diff(bins), color=colors, label='Color')
    sm = ScalarMappable(Normalize(vmin=min(data), vmax=max(data)), cmap='viridis')
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', shrink=0.3)
    cbar.set_label('Vegetation Index Value')

    # Set the title and labels
    name_title = f'Plot{plot_index}_{title}\nTh= {optimal_threshold:.2f}'
    ax.set_title(name_title)
    ax.set_xlabel('Vegetation Index Value')
    ax.set_ylabel('Frequency')

    # Save the plot as an image file
    plt.savefig(output_path)
    plt.close(fig)  # Close the figure to free memory

###########################RGB#######################################
class RGB2Dataset:

    def __init__(self,src,gdf,output_dir,filename):
        self.src = src
        self.gdf = gdf
        self.output_dir = output_dir
        self.filename = filename


    @staticmethod
    def calculate_canopy_cover(input_dir):
        VI_selected = input_dir['vi']
        selected_threshold = input_dir['threshold']
        title = input_dir['title']
        red = input_dir['red']
        green = input_dir['green']
        blue = input_dir['blue']

        # Apply thresholds to create a mask
        mask = VI_selected >= selected_threshold if title in ['ExR', 'CIVE'] else VI_selected < selected_threshold
        VI_masked = np.where(mask, np.nan, VI_selected)  # Mask the non-vegetation pixels

        vegetation_pixels = np.sum(~np.isnan(VI_masked))  # Extract the vegetation pixels
        total_pixels = np.prod(VI_masked.shape) - np.sum(np.isnan(VI_selected))  # Total pixels
        vegetation_cover_ratio = vegetation_pixels / total_pixels  # Canopy cover

        red_masked = np.where(mask, np.nan, red)
        green_masked = np.where(mask, np.nan, green)
        blue_masked = np.where(mask, np.nan, blue)

        red_mean = np.nanmean(red_masked)
        green_mean = np.nanmean(green_masked)
        blue_mean = np.nanmean(blue_masked)

        return {'cc': vegetation_cover_ratio, 'red': red_mean, 'green': green_mean, 'blue': blue_mean}
    

    @staticmethod
    def calculate_vi(red, green, blue, selected_vi):
        # Define the VI calculations
        def ndi_calc(red, green): return 128 * (((green - red) / (green + red)) + 1)
        def exg_calc(red, green, blue): return 2 * green - red - blue
        def exr_calc(red, green): return 1.3 * red - green
        def cive_calc(red, green, blue): return 0.441 * red - 0.811 * green + 0.385 * blue + 18.78745
        def veg_calc(red, green, blue): return green / (red**0.667 * blue**(1 - 0.667))

        # Calculate all VIs if no specific VI is selected
        if selected_vi is None:
            exg = exg_calc(red, green, blue)
            exr = exr_calc(red, green)
            ndi = ndi_calc(red, green)
            cive = cive_calc(red, green, blue)
            exgr = exg - exr
            veg = veg_calc(red, green, blue)
            com1 = exg + cive + exgr + veg
            mexg = 1.262 * green - 0.884 * red - 0.311 * blue
            com2 = 0.36 * exg + 0.47 * cive + 0.17 * veg

            VIs = [('NDI', ndi), ('ExG', exg), ('ExR', exr), ('CIVE', cive), ('ExGR', exgr), ('COM1', com1), ('MExG', mexg), ('COM2', com2), ('VEG', veg)]
            return VIs
        else:
            # Calculate the selected VI
            vi_calculations = {
                'NDI': ndi_calc(red, green),
                'ExG': exg_calc(red, green, blue),
                'ExR': exr_calc(red, green),
                'CIVE': cive_calc(red, green, blue),
                'ExGR': exg_calc(red, green, blue) - exr_calc(red, green),
                'COM1': exg_calc(red, green, blue) + cive_calc(red, green, blue) + (exg_calc(red, green, blue) - exr_calc(red, green)) + veg_calc(red, green, blue),
                'MExG': 1.262 * green - 0.884 * red - 0.311 * blue,
                'COM2': 0.36 * exg_calc(red, green, blue) + 0.47 * cive_calc(red, green, blue) + 0.17 * veg_calc(red, green, blue),
                'VEG': veg_calc(red, green, blue)
            }
            return vi_calculations[selected_vi]
    
    ## function 1: Clip image.
    # Clipping raw data based on the shape file boundry and store them in desired locations
    
    def clip_rasterio_shape(self):
        """
            Input: 
                image_path: path to image
                shapefile_path: path to your shapefile
        """
        
        # Create output directory
        output_path = self.output_dir +'/Shape clip_'+ f'{self.filename}'

        if not os.path.exists(output_path):
            os.makedirs(output_path)


        # Clip image to the shapefile geometry
        clipped_image, clipped_transform = mask(self.src, self.gdf.geometry, crop=True)

        # Specify output path for each individual band using the image name
        clipped_image_path = os.path.join(output_path, f'{self.filename}-clipped.tif')

        # Create a new raster file with the same dimensions as the clipped band
        with rasterio.open(clipped_image_path, 'w', driver='GTiff', width=clipped_image.shape[2], height=clipped_image.shape[1], count=4, dtype=clipped_image.dtype, crs=self.src.crs, transform=clipped_transform) as dst:
            dst.write(clipped_image)
        print(f'clipped image {self.filename}-clipped.tif saved to {clipped_image_path}')
        
        # Print shape and CRS (sanity check)
        print(f'Shape: {clipped_image.shape}, CRS: {self.src.crs}')
        
        # loop through the selected polygons and save each clipped image separately
        for polygon_idx, polygon in enumerate(self.gdf.geometry):
            # Extract the geometry of the selected polygon
            polygon_geometry = self.gdf.iloc[polygon_idx]['geometry']
            plot_name = self.gdf.iloc[polygon_idx]['Plot_ID']   # here we cannot fix it, should be changable.

            # Clip the image to the selected polygon's geometry and specify the output CRS
            clipped_image, clipped_transform = mask(self.src, [polygon_geometry], crop=True)
            clipped_meta = self.src.meta.copy()
            clipped_meta['crs'] = {'init': f'epsg:{self.gdf.crs}'}

            # output file path for the clipped image using the band and location index
            output_image_path = os.path.join(output_path, f'{plot_name}.tif')

            # Create a new raster file with the same dimensions as the clipped image
            with rasterio.open(output_image_path, 'w', driver='GTiff', width=clipped_image.shape[2], height=clipped_image.shape[1], count=clipped_image.shape[0], dtype=clipped_image.dtype, crs=self.src.crs, transform=clipped_transform) as dst:
                dst.write(clipped_image)

        print("Clipped images saved in the output directories for each plot.")
        # Return the path to the clipped image
        return send_file(clipped_image_path, as_attachment=True)


    def dataset_extraction_auto(self,target_df):
        
        # Create output directory
        output_path = self.output_dir +'/Excel_'+ f'{self.filename}'

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        plot_num = self.gdf.shape[0]
        cc_all = [[None] * 9 for _ in range(plot_num)]
        red_all =[[None] * 9 for _ in range(plot_num)]
        green_all =[[None] * 9 for _ in range(plot_num)]
        blue_all = [[None] * 9 for _ in range(plot_num)]

        # with rasterio.open(image_path) as src:
        # loop through the selected polygons and save each clipped image separately
        for polygon_idx, polygon in enumerate(self.gdf.geometry):

            # Extract the geometry of the selected polygon
            polygon_geometry = self.gdf.iloc[polygon_idx]['geometry']

            # Clip the image to the selected polygon's geometry and specify the output CRS
            clipped_image, clipped_transform = mask(self.src, [polygon_geometry], crop=True)

            # clear out the background fill
            mask_black = (clipped_image <=0)
            masked_img = np.where(mask_black,np.nan, clipped_image)

            red = masked_img[0,:,:]
            green = masked_img[1,:,:]
            blue = masked_img[2,:,:]
            

            # VI(vegetation index)
            VIs = RGB2Dataset.calculate_vi(red, green, blue,None)
            cc =[]
            red_mean =[]
            green_mean =[]
            blue_mean = []

            for (title, vi) in VIs:
                vi_values = vi.flatten() # Flatten the image array to a 1D array
                vi_values = vi_values[~np.isnan(vi_values)]  # keep the non-NA vi values
                threshold = otsu(vi_values,False)     # calculate thresholds 
                var_tuple ={'title': title,'vi':vi,'threshold':threshold,'red':red,'green':green, 'blue':blue}
                output = RGB2Dataset.calculate_canpoy_cover(var_tuple)  # calculate the vegetation ratio
                cc.append(output['cc'])
                red_mean.append(output['red'])
                green_mean.append(output['green'])
                blue_mean.append(output['blue'])

            cc_all[polygon_idx]= cc
            red_all[polygon_idx]= red_mean
            green_all[polygon_idx]= green_mean
            blue_all[polygon_idx]= blue_mean

        # Save all cc to a dataframe
        col_name = ['NDI','ExG','ExR','CIVE','ExGR','COM1','MExG','COM2','VEG']    
        df = pd.DataFrame(cc_all, columns= col_name)
        df.insert(0,column='Plot_ID',value=self.gdf['Plot_ID'])
        
        # merge the yld dataframe and cc_all to make sure they are in the same order
        merged_df = pd.merge(df,target_df,how='inner', left_on='Plot_ID',right_on='Plot_ID')
        print(merged_df.head())

        if merged_df.empty: # if it is empty, maybe the name of the plot_id is different
           # Change to HTML message
           print(f"You may need to name your Plot_ID column by this format:{self.gdf['Plot_ID']}")
           return

        # calculate the correlation and find max one.(It will write all of them if the max are more than one)
        target_col_name = merged_df.columns[-1]
        correlations = {}
        for col in merged_df.columns[1:-1]:
            correlation = merged_df[col].corr(merged_df[target_col_name]) # last column is target 
            correlations[col] = correlation
        print(correlations)    

        # Save the cc has highest correlation with yield to new dataframe
            
        max_corr_value = correlations.get(max(correlations))   # max corr value
        max_corr_columns = [col for col, correlation in correlations.items() if correlation == max_corr_value] # all columns name with max corr values
        round_max_corr = r'{:.2f}'.format(float(max_corr_value))
        print(max_corr_columns,max_corr_value)

        for col in max_corr_columns:
            df_new = merged_df.iloc[:,:1]
            df_new[target_col_name] = merged_df[target_col_name]
            df_new['CC'] = merged_df[f'{col}']

            max_corr_index = merged_df.columns.get_loc(col) -2
            print(max_corr_index)
            red = np.array([row[max_corr_index] for row in red_all]).astype(float)
            green = np.array([row[max_corr_index] for row in green_all]).astype(float)
            blue = np.array([row[max_corr_index] for row in blue_all]).astype(float)
            
            VIs = RGB2Dataset.calculate_vi(red,green,blue,None)
            data_df = [('red',red),('green',green),('blue',blue)] + VIs
        
            # dataframe
            for colname,coldata in data_df:
                df_new[colname] = coldata

            # write to the excel
            excel_filepath = os.path.join(output_path, f'dataset_{col}_{round_max_corr}.xlsx')
            df_new.to_excel(excel_filepath, index=False)

        print('Dataset saved!!')
        # Return the path to the Excel file
        return send_file(excel_filepath, as_attachment=True)
    
    # Test: select a plot to see
    def visualization_plot(self, plot_index, hist_or_cc):

        # Extract the geometry of the selected polygon
        polygon_geometry = self.gdf.iloc[plot_index]['geometry']
        plot_name = self.gdf.iloc[plot_index]['Plot_ID']

        # Clip the image to the selected polygon's geometry and specify the output CRS
        clipped_image, clipped_transform = mask(self.src, [polygon_geometry], crop=True)

        # mask balck background
        mask_black = (clipped_image <=0)
        masked_img = np.where(mask_black,np.nan, clipped_image)
        red= masked_img[0,:,:]
        green = masked_img[1,:,:]
        blue = masked_img[2,:,:]

        # VI(vegetation index)
        VIs = RGB2Dataset.calculate_vi(red,green,blue,None)

        fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 8))
        axs = np.ravel(axs)
        if hist_or_cc:
            for ax, (title, vi) in zip(axs, VIs):
                vi_values = vi.flatten() # Flatten the image array to a 1D array
                vi_values = vi_values[~np.isnan(vi_values)]
                plot_VI_hist(ax,title,vi_values,plot_name)     # Plot histogram of pixel values

            plt.suptitle(f'{plot_name}:vegetation index hist')
            plt.tight_layout()
            plt.show()

        else:
            threshold_all = []

            for (title, vi) in VIs:
                vi_values = vi.flatten() # Flatten the image array to a 1D array
                vi_values = vi_values[~np.isnan(vi_values)]  # keep the non-NA vi values
                threshold = otsu(vi_values,False)     # calculate thresholds 
                threshold_all.append(threshold)

            VIs_th = [(v[0],v[1],t) for v,t in zip(VIs, threshold_all)]

            for ax, (title, vi, threshold) in zip(axs, VIs_th):
                VI_selected = vi
                threshold_selected = threshold

                # Apply thresholds to create a mask
                mask_selected =  VI_selected < threshold_selected

                if title == 'ExR' or title == "CIVE":
                    mask_selected =  VI_selected >= threshold_selected

                VI_masked = np.where(mask_selected,np.nan,VI_selected) # mask the non veg pixels
                vegetation_pixels = np.sum(~np.isnan(VI_masked)) # extrat the veg pixels
                total_pixels = np.prod(VI_masked.shape)-np.sum(np.isnan(VI_selected)) # the total pixels
                vegetation_cover_ratio = vegetation_pixels / total_pixels  # canopy cover


                # Plot vegetaion pixels on original image
                pure_soil_pixel_coords = [(col, row) for row, col in zip(*np.where(~np.isnan(VI_masked)))]
                crs_coords = [(col, row) for col,row in pure_soil_pixel_coords]     # Get the CRS coordinates of the pure pixel coordinates

                # Plot the scatter plot for the pure pixel coordinates in CRS coordinates
                crs_x_coords, crs_y_coords = zip(*crs_coords)
                ax.scatter(crs_x_coords, crs_y_coords, c='red', marker='x', s=1, label='Pure veg Pixels')        

                # # Display image and Set the labels and colorbar for the image
                # extent = (self.src.bounds.left, self.src.bounds.right, self.src.bounds.bottom, self.src.bounds.top)
                ax.imshow(clipped_image.transpose(1,2,0), cmap='RdYlGn', aspect=1)
                ax.axis('off')
                ax.set_title(f'{title} : CC={vegetation_cover_ratio}')


            # Save the figure to a file instead of showing it
            fig_path = os.path.join(self.output_dir, f'visualization_plot_{plot_index}.png')
            plt.savefig(fig_path)
            plt.close()  # Close the figure to free memory

            # Return the path to the figure
            return fig_path
    
    def visualization_shpfile(self):
        ax = self.gdf.geometry.plot(figsize =(12,8))
        self.gdf.apply(lambda x: ax.annotate(text=x['Plot_ID'], xy=x.geometry.centroid.coords[0], ha='center'), axis=1)
        # Save the figure to a file instead of showing it
        fig_path = os.path.join(self.output_dir, 'visualization_shpfile.png')
        plt.savefig(fig_path)
        plt.close()  # Close the figure to free memory

        # Return the path to the figure
        return fig_path

    def check_original(self):
        # Sanity check for proper clipping (by merging shape file on the top of the stacked images)

        # Create a Geodata-frame with the clipped geometry
        gdf_clipped = gpd.GeoDataFrame({'geometry': [g for g in self.gdf.geometry]}, crs=self.gdf.crs)


        # Plot each band image with the CRS coordinates and their band names
        image = self.src.read()
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        im = rasterio.plot.show(image, transform=self.src.transform, ax=ax, cmap='viridis')  # You can specify a colormap (e.g., 'viridis')
        gdf_clipped.boundary.plot(ax=ax, color='red')  # Add the shapefile boundaries for reference
        self.gdf.apply(lambda x: ax.annotate(text=x['Plot_ID'], xy=x.geometry.centroid.coords[0], ha='center'), axis=1)
        ax.set_title(f"Band {self.filename} in CRS Coordinates")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        # Save the figure to a file instead of showing it
        fig_path = os.path.join(self.output_dir, 'check_original.png')
        plt.savefig(fig_path)
        plt.close()  # Close the figure to free memory

        # Return the path to the figure
        return fig_path

    def check_clipped(self):
        # src = rasterio.open(clipped_image_path)
        clipped_image, clipped_transform = mask(self.src, self.gdf.geometry, crop=True)

        # Create a Geodata-frame with the clipped geometry
        gdf_clipped = gpd.GeoDataFrame({'geometry': [g for g in self.gdf.geometry]}, crs=self.gdf.crs)


        # Plot each band image with the CRS coordinates and their band names
        image = clipped_image
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # mask_black = image <=0
        # image = np.where(mask_black,np.nan,image)
        im = rasterio.plot.show(image, transform= clipped_transform, ax=ax, cmap='viridis')  # You can specify a colormap (e.g., 'viridis')
        # ax.imshow(image.transpose(1,2,0),cmap='viridis')
        gdf_clipped.boundary.plot(ax=ax, color='red')  # Add the shapefile boundaries for reference
        self.gdf.apply(lambda x: ax.annotate(text=x['Plot_ID'], xy=x.geometry.centroid.coords[0], ha='center',color ='red'), axis=1)
        ax.set_title(f"{self.filename} in CRS Coordinates")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        # Save the figure to a file instead of showing it
        fig_path = os.path.join(self.output_dir, 'check_clipped.png')
        plt.savefig(fig_path)
        plt.close()  # Close the figure to free memory

        # Return the path to the figure
        return fig_path

    # calculate a datalist    
    def dataset_extraction_manu_threshold(self,selected_vi):
        data_for_window = []
        
        # loop through the selected polygons and save each clipped image separately
        for polygon_idx, polygon in enumerate(self.gdf.geometry):

            # Extract the geometry of the selected polygon
            polygon_geometry = self.gdf.iloc[polygon_idx]['geometry']
            plot_name = self.gdf.iloc[polygon_idx]['Plot_ID']

            # Clip the image to the selected polygon's geometry and specify the output CRS
            clipped_image, clipped_transform = mask(self.src, [polygon_geometry], crop=True)

            # clear out the background fill
            mask_black = (clipped_image <=0)
            masked_img = np.where(mask_black,np.nan, clipped_image)

            red = masked_img[0,:,:]
            green = masked_img[1,:,:]
            blue = masked_img[2,:,:]

            vi = RGB2Dataset.calculate_vi(red,green,blue,selected_vi)
            vi_values = vi.flatten() # Flatten the image array to a 1D array
            vi_values = vi_values[~np.isnan(vi_values)]  # keep the non-NA vi values
            threshold = otsu(vi_values,False)     # calculate thresholds 
            # [index,plot_Id,otsu_th,min_th,max_th] 
            list_data = [polygon_idx,plot_name,threshold,min(vi_values),max(vi_values)]
            data_for_window.append(list_data)

        return data_for_window    
     
    # visualization: for one plot, one hist and one cc
    def dataset_extraction_manu_plot(self,selected_vi,threshold,plot_index, hist_or_cc, output_dir):

        # Extract the geometry of the selected polygon
        polygon_geometry = self.gdf.iloc[plot_index]['geometry']
        plot_name = self.gdf.iloc[plot_index]['Plot_ID']

        # Clip the image to the selected polygon's geometry and specify the output CRS
        clipped_image, clipped_transform = mask(self.src, [polygon_geometry], crop=True)

        # mask balck background
        mask_black = (clipped_image <=0)
        masked_img = np.where(mask_black,np.nan, clipped_image)
        red= masked_img[0,:,:]
        green = masked_img[1,:,:]
        blue = masked_img[2,:,:]

        # VI(vegetation index)
        vi = RGB2Dataset.calculate_vi(red,green,blue,selected_vi)

        title = selected_vi
        fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        if hist_or_cc:
            vi_values = vi.flatten() # Flatten the image array to a 1D array
            vi_values = vi_values[~np.isnan(vi_values)]
            plot_VI_hist(ax,title,vi_values,plot_name)     # Plot histogram of pixel values
            plt.suptitle(f'{plot_name}:vegetation index hist')
            plt.tight_layout()
            plt.show()

        else:
            VI_selected = vi
            threshold_selected = threshold

            # Apply thresholds to create a mask
            mask_selected =  VI_selected < threshold_selected
            if title == 'ExR' or title == "CIVE":
                mask_selected =  VI_selected >= threshold_selected

            VI_masked = np.where(mask_selected,np.nan,VI_selected) # mask the non veg pixels
            vegetation_pixels = np.sum(~np.isnan(VI_masked)) # extrat the veg pixels
            total_pixels = np.prod(VI_masked.shape) -np.sum(np.isnan(VI_selected)) # the total pixels
            vegetation_cover_ratio = vegetation_pixels / total_pixels  # canopy cover


            # Plot vegetaion pixels on original image
            pure_soil_pixel_coords = [(col, row) for row, col in zip(*np.where(~np.isnan(VI_masked)))]
            crs_coords = [(col, row) for col,row in pure_soil_pixel_coords]     # Get the CRS coordinates of the pure pixel coordinates

            # Plot the scatter plot for the pure pixel coordinates in CRS coordinates
            crs_x_coords, crs_y_coords = zip(*crs_coords)
            ax.scatter(crs_x_coords, crs_y_coords, c='red', marker='x', s=1, label='Pure veg Pixels')  
   

            # # Display image and Set the labels and colorbar for the image
            # extent = (self.src.bounds.left, self.src.bounds.right, self.src.bounds.bottom, self.src.bounds.top)
            ax.imshow(clipped_image.transpose(1,2,0), cmap='RdYlGn', aspect=1)
            ax.axis('off')
            ax.set_title(f'{title} : CC={vegetation_cover_ratio}')

            # Save the figure to a file instead of showing it
            fig_path = os.path.join(output_dir, f'plot_{plot_index}_{selected_vi}.png')
            plt.savefig(fig_path)
            plt.close()
            return fig_path

    # need to calculate all and save all based on readed parameters
    def dataset_extraction_manu_ok(self,selected_vi, selected_threshold, output_dir):
        # need to calculate cc, red_mean,green_mean,blue_mean,vi
        # Create output directory
        output_path = self.output_dir +'/Excel_'+ f'{self.filename}'

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        pass

        cc =[]
        red_mean =[]
        green_mean =[]
        blue_mean = []

        # with rasterio.open(image_path) as src:
        # loop through the selected polygons and save each clipped image separately
        for polygon_idx, polygon in enumerate(self.gdf.geometry):

            # Extract the geometry of the selected polygon
            polygon_geometry = self.gdf.iloc[polygon_idx]['geometry']

            # Clip the image to the selected polygon's geometry and specify the output CRS
            clipped_image, clipped_transform = mask(self.src, [polygon_geometry], crop=True)

            # clear out the background fill
            mask_black = (clipped_image <=0)
            masked_img = np.where(mask_black,np.nan, clipped_image)

            red = masked_img[0,:,:]
            green = masked_img[1,:,:]
            blue = masked_img[2,:,:]


            title = selected_vi
            # VI(vegetation index)
            vi = RGB2Dataset.calculate_vi(red,green,blue,selected_vi)
            threshold = selected_threshold[polygon_idx]
            var_tuple ={'title': title,'vi':vi,'threshold':threshold,'red':red,'green':green, 'blue':blue}
            output = RGB2Dataset.calculate_canpoy_cover(var_tuple)  # calculate the vegetation ratio

            cc.append(output['cc'])
            red_mean.append(output['red'])
            green_mean.append(output['green'])
            blue_mean.append(output['blue'])
        
        VIs = RGB2Dataset.calculate_vi(np.array(red_mean).astype(float),np.array(green_mean).astype(float),np.array(blue_mean).astype(float),None)
        data_list = [('Plot_ID',self.gdf['Plot_ID']),(f'CC_{selected_vi}',cc),('red',red_mean),('green',green_mean),('blue',blue_mean)]+ VIs
        data_dict ={col_name: data for col_name, data in data_list } 
 
        df = pd.DataFrame(data_dict)

        # Save the Excel file to the output directory
        excel_filepath = os.path.join(output_dir, f'dataset_{selected_vi}.xlsx')
        df.to_excel(excel_filepath, index=False)
        print('Dataset saved!!')
        return excel_filepath   
