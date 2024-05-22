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


def otsu(data, is_plot):
    """
    Input: 
        data: vegetation index data. 
        is_plot: if it is true, will return hist,bins,and optimal_threshold; otherwise, only need to return optimal_threshold.
    
    """
    hist,bins = np.histogram(data,bins= 100,range=(np.min(data),np.max(data)))

    ## ostu method
    total_freq = sum(hist)
    
    # Compute probabilities of each intensity level
    probabilities =  hist/ total_freq

    # Initialize variables for Ostu's method
    max_variance = 0
    optimal_threshold = 0

    for t in range(1, len(bins)):
        w0 = np.sum(probabilities [:t])
        w1 = np.sum(probabilities [t:])

        if 0 == w0:
            break

        if 0 == w1:
            continue
        
        mean0 = np.sum(probabilities [:t] * bins[:t])/w0
        mean1 = np.sum(probabilities [t:] * bins[t:-1])/w1

        variance = w0 *w1 *(mean0-mean1) **2

        if variance >max_variance:
            max_variance = variance
            optimal_threshold = bins[t]

    if is_plot ==True:
        return hist,bins,optimal_threshold
    else:
        return optimal_threshold

def plot_VI_hist(ax,title,data,plot_index):

    hist,bins,optimal_threshold= otsu(data,True)
    # plot the hist of VI
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i/len(hist)) for i in range(len(hist))]

    if title == 'ExR' or title =='CIVE':
        colors = colors[::-1]

    ax.bar(bins[:-1], hist,width=np.diff(bins),color = colors, label ='Color')
    sm = ScalarMappable(cmap='viridis')
    sm.set_array([data])
    cbar = plt.colorbar(sm, ax = ax,orientation='vertical',shrink=0.3)
    cbar.set_label('Vegetation Index Value')

    # Anzeigen der Werte in einer Box
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    name_title = f'Plot{plot_index}_{title}' +'\n'+ f'Th= {optimal_threshold:.2f} '
    ax.legend([extra], [name_title], loc='upper left')

    # ax.set_title()
    ax.set_xlabel('Vegetation Index Value')
    ax.set_ylabel('Frequency')


###########################RGB#######################################
class RGB2Dataset:

    def __init__(self,src,gdf,output_dir,filename):
        self.src = src
        self.gdf = gdf
        self.output_dir = output_dir
        self.filename = filename
        
    @staticmethod
    def calculate_canpoy_cover(input_dir):
        """
        Input parameters:input_dir is a dictionary(key-value pair)
            input_dir: 'vi', VI values
                       'threshold': calculated threshold from ostu
                       'title': vi name
                       'red':red band pixel values
                       'green': green band pixel values
                       'blue':blue band pixel values
        return: 
            'cc':vegetation_cover_ratio
            'red':red_mean
            'green':green_mean
            'blue':blue_mean         
        """
        
        VI_selected = input_dir['vi']
        selected_threshold = input_dir['threshold']
        title = input_dir['title']
        red = input_dir['red']
        green = input_dir['green']
        blue = input_dir['blue']

        # Apply thresholds to create a mask
        if title == 'ExR' or title == "CIVE":
            mask =  VI_selected >= selected_threshold
        else:     
            mask =  VI_selected < selected_threshold 

        VI_masked = np.where(mask,np.nan,VI_selected) # mask the non veg pixels

        vegetation_pixels = np.sum(~np.isnan(VI_masked)) # extrat the veg pixels

        total_pixels = np.prod(VI_masked.shape)-np.sum(np.isnan(VI_selected)) # the total pixels

        vegetation_cover_ratio = vegetation_pixels / total_pixels  # canopy cover

        red_masked = np.where(mask,np.nan,red)
        green_masked = np.where(mask,np.nan,green)
        blue_masked = np.where(mask,np.nan,blue)

        red_masked = red_masked[~np.isnan(red_masked)]
        green_masked = green_masked[~np.isnan(green_masked)]
        blue_masked = blue_masked[~np.isnan(blue_masked)]

        red_mean = np.mean(red_masked)
        green_mean = np.mean(green_masked)
        blue_mean =np.mean(blue_masked)

        return {'cc':vegetation_cover_ratio,'red':red_mean, 'green':green_mean,'blue':blue_mean}

    @staticmethod
    def calculate_vi(red,green,blue,selected_vi):

        if selected_vi == None:

            exg = 2*green - red - blue
            exr = 1.3*red -green
            ndi = 128*(((green-red)/(green +red))+1)
            cive = 0.441*red-0.811*green+0.385*blue +18.78745 
            exgr = exg -exr
            veg = green/(red**0.667 * blue**(1-0.667))
            com1 = exg + cive + exgr + veg
            mexg = 1.262 *green -0.884*red -0.311*blue
            com2 = 0.36 *exg + 0.47*cive + 0.17*veg

            VIs = [('NDI' , ndi),('ExG' , exg), ('ExR' , exr), ('CIVE' , cive), ('ExGR' , exgr), ('COM1', com1), ('MExG' , mexg), ('COM2' , com2),('VEG' , veg)]
            
            return  VIs
        else:

            match selected_vi:
                case 'NDI':
                    vi = 128*(((green-red)/(green +red))+1)
                case 'ExG':
                    vi = 2*green - red - blue
                case 'ExR':
                    vi = 1.3*red -green
                case 'CIVE':
                    vi =0.441*red-0.811*green+0.385*blue +18.78745
                case 'ExGR':
                    exg = 2*green - red - blue
                    exr = 1.3*red -green
                    vi = exg -exr
                case 'COM1':
                    exg = 2*green - red - blue
                    exr = 1.3*red -green
                    cive = 0.441*red-0.811*green+0.385*blue +18.78745
                    exgr = exg -exr
                    veg = green/(red**0.667 * blue**(1-0.667))
                    vi= exg + cive + exgr + veg 
                case 'MExG':
                     vi =1.262 *green -0.884*red -0.311*blue
                case 'COM2':
                     vi = 0.36 *exg + 0.47*cive + 0.17*veg
                case 'VEG':
                    vi = green/(red**0.667 * blue**(1-0.667))
            return vi        
            
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


            # axs[-1].imshow(clipped_image.transpose(1,2,0), cmap='RdYlGn', aspect=1)
            # axs[-1].axis('off')
            plt.suptitle(f'{plot_name}:vegetation pixels')
            plt.tight_layout()
            plt.show()
        return
 
    def visualization_shpfile(self):
        ax = self.gdf.geometry.plot(figsize =(12,8))
        self.gdf.apply(lambda x: ax.annotate(text=x['Plot_ID'], xy=x.geometry.centroid.coords[0], ha='center'), axis=1)

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

        plt.tight_layout()
        plt.show()  

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

        plt.tight_layout()
        plt.show()       

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
    def dataset_extraction_manu_plot(self,selected_vi,threshold,plot_index, hist_or_cc):

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

            plt.suptitle(f'{plot_name}:vegetation pixels with threshold {threshold}')
            plt.tight_layout()
            plt.show()

    # need to calculate all and save all based on readed parameters
    def dataset_extraction_manu_ok(self,selected_vi, selected_threshold):
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

        # write to the excel
        excel_filepath = os.path.join(output_path, f'dataset_{selected_vi}.xlsx')
        df.to_excel(excel_filepath, index=False)

        print('Dataset saved!!')            

############################ MS ######################################
# Mutispectral image class:
class MS2Dataset():
    def __init__(self,src_r,src_g,src_b,src_re,src_nir,gdf,output_dir,is_stacked):
        self.src_r = src_r
        self.src_g = src_g
        self.src_b = src_b
        self.src_re = src_re
        self.src_nir = src_nir
        self.gdf = gdf
        self.output_dir = output_dir
        self.is_stacked = is_stacked


    def mask_black(self,polygon_geometry):
            # Clip the image to the selected polygon's geometry and specify the output CRS
        clipped_image_r, clipped_transform_r = mask(self.src_r, [polygon_geometry], crop=True)
        clipped_image_g, clipped_transform_g = mask(self.src_g, [polygon_geometry], crop=True)
        clipped_image_b, clipped_transform_b = mask(self.src_b, [polygon_geometry], crop=True)
        clipped_image_re, clipped_transform_re = mask(self.src_re, [polygon_geometry], crop=True)
        clipped_image_nir, clipped_transform_nir = mask(self.src_nir, [polygon_geometry], crop=True)


        clipped_images=[('red',clipped_image_r),('green',clipped_image_g),('blue',clipped_image_b),('rededge',clipped_image_re),('nir',clipped_image_nir)]

        # clear out the background fill
        for name,clipped_image in clipped_images:
            mask_black = (clipped_image<=0)
            masked_img = np.where(mask_black,np.nan, clipped_image)
            if name == 'red':
                red = masked_img[0,:,:]
            elif name =='green':
                green = masked_img[0,:,:]
            elif name == 'blue':
                blue = masked_img[0,:,:]
            elif name=='rededge':
                rededge = masked_img[0,:,:]
            elif name =='nir':
                nir =masked_img[0,:,:] 

        return red,green,blue,rededge,nir           

    @staticmethod
    def calculate_canpoy_cover(input_dir):
        
        VI_selected = input_dir['vi']
        selected_threshold = input_dir['threshold']
        # title = input_dir['title']
        red = input_dir['red']
        green = input_dir['green']
        blue = input_dir['blue']
        rededge = input_dir['re']
        nir = input_dir['nir']
        # Apply thresholds to create a mask
        mask =  VI_selected < selected_threshold

        #  if title == 'ExR' or title == "CIVE":
        #       mask =  VI_selected >= selected_threshold

        VI_masked = np.where(mask,np.nan,VI_selected) # mask the non veg pixels

        vegetation_pixels = np.sum(~np.isnan(VI_masked)) # extrat the veg pixels

        total_pixels = np.prod(VI_masked.shape)-np.sum(np.isnan(VI_selected)) # the total pixels

        vegetation_cover_ratio = vegetation_pixels / total_pixels  # canopy cover

        red_masked = np.where(mask,np.nan,red)
        green_masked = np.where(mask,np.nan,green)
        blue_masked = np.where(mask,np.nan,blue)
        rededge_masked = np.where(mask,np.nan,rededge)
        nir_masked = np.where(mask,np.nan,nir)

        red_masked = red_masked[~np.isnan(red_masked)]
        green_masked = green_masked[~np.isnan(green_masked)]
        blue_masked = blue_masked[~np.isnan(blue_masked)]
        rededge_masked = rededge_masked[~np.isnan(rededge_masked)]
        nir_masked = nir_masked[~np.isnan(nir_masked)]

        red_mean = np.mean(red_masked)
        green_mean = np.mean(green_masked)
        blue_mean =np.mean(blue_masked)
        rededge_mean =np.mean(rededge_masked)
        nir_mean =np.mean(nir_masked)

        return {'cc':vegetation_cover_ratio,'red':red_mean,'green':green_mean,'blue':blue_mean,'re':rededge_mean,'nir':nir_mean}

    @staticmethod
    def calculate_VI(red,green,blue,rededge,nir,selected_vi):
        # VI(vegetation index)
        # Constants for EVI calculation
        L = 0.16  # Canopy background adjustment; gsavi
        x=0.08 # atsavi
        a=1.43 # atsavi, savi2
        b=0.01 # atsavi, savi2

        # Calculate VIs
        if selected_vi != None:
            match selected_vi:
                case 'NDVI':
                    vi = (nir - red) / (nir + red)
                case 'GNDVI':
                    vi = (nir - green) / (nir + green)
                case 'DVI':
                    vi = (nir - red)
                case 'EVI':
                    vi = (2.5*(nir-red))/((nir+(6*red)-(7.5*blue))+1) #evi
                case 'EVI2':
                    vi = (2.5*(nir-red))/((nir+(2.4*red))+1)
                case 'ARI':
                    vi = (1/green)-(1/rededge)
                case 'MARI':
                    vi = (((1/green)-(1/rededge))*nir)
                case 'CHLRE': 
                    vi = (nir/rededge)-1
                case 'CHLGR':
                    vi = (nir/green)-1 
                case 'SAVI':
                    savi= (1.5*(nir - red))/(nir+red+0.5) # SAVI
                case 'SAVI2':
                    savi2= nir/(red+(b/a)) # SAVI2
                case 'MSAVI':
                    vi= ((2*nir)+1-np.sqrt(((2*nir)+1)**2-(8*(nir-red))))/2 # MSAVI
                case 'OSAVI':
                    vi=(1+L)*((nir-green)/(nir+green+L)) #OSAVI 
                case 'TSAVI':
                    vi = ((a*(nir-(a*red)-b))/((a*nir)+red-(a*b)+(x*(1+a**2)))) #or TSAVI
                case 'MSR':
                    vi = ((nir/red)-1)/np.sqrt(((nir/red)-1))
                case 'MCARI': 
                    vi = ((rededge - red) - 0.2 * (rededge - green)) * (rededge / red)
                case 'MACRIDIOSAVI':
                    osavi = (1+L)*((nir-green)/(nir+green+L))
                    mcari = ((rededge - red) - 0.2 * (rededge - green)) * (rededge / red)
                    vi = mcari/osavi
                case 'PVI':
                    vi = (nir-(a*red)-b)/(np.sqrt(1+a**2))
                case 'SR':
                    vi = nir/red
                case 'WDRI': 
                    vi = (0.3*(nir-red))/(0.3*(nir+red))   
            return vi                       

        else:
            ndvi = (nir - red) / (nir + red) # ndvi
            gndvi =(nir - green) / (nir + green)
            dvi= (nir - red)
            evi = (2.5*(nir-red))/((nir+(6*red)-(7.5*blue))+1) #evi
            evi2 = (2.5*(nir-red))/((nir+(2.4*red))+1)
            ari= (1/green)-(1/rededge)
            mari = (((1/green)-(1/rededge))*nir)
            chlre= (nir/rededge)-1
            chlgr= (nir/green)-1 
            savi= (1.5*(nir - red))/(nir+red+0.5) # SAVI
            savi2= nir/(red+(b/a)) # SAVI2
            msavi= ((2*nir)+1-np.sqrt(((2*nir)+1)**2-(8*(nir-red))))/2 # MSAVI
            osavi=(1+L)*((nir-green)/(nir+green+L)) #OSAVI 
            tsavi = ((a*(nir-(a*red)-b))/((a*nir)+red-(a*b)+(x*(1+a**2)))) #or TSAVI
            msr = ((nir/red)-1)/np.sqrt(((nir/red)-1))
            mcari = ((rededge - red) - 0.2 * (rededge - green)) * (rededge / red)
            mcaridiosavi = mcari/osavi
            pvi = (nir-(a*red)-b)/(np.sqrt(1+a**2))
            sr=nir/red
            wdri= (0.3*(nir-red))/(0.3*(nir+red))

            VIs = [('NDVI',ndvi), ('GNDVI',gndvi), ('DVI',dvi), ('EVI',evi), ('EVI2',evi2), ('ARI',ari),('MARI',mari), ('CHLRE',chlre), ('CHLGR',chlgr), 
                ('SAVI',savi), ('SAVI2',savi2), ('MSAVI',msavi), ('OSAVI',osavi), ('TSAVI',tsavi), ('MSR',msr), ('MCARI',mcari), 
                ('MCARIDIOSAVI',mcaridiosavi), ('PVI',pvi), ('SR',sr), ('WDRI',wdri)]
            
            return VIs

    # clip shape for each band
    def clip_rasterio_shape(self):
        """
        Input: 
            image_path: path to image
            shapefile_path: path to your shapefile
        """
        
        # Create output directory
        output_path_clip = self.output_dir +'/Shape clip/'

        if not os.path.exists(output_path_clip):
            os.makedirs(output_path_clip)

        # all_src =[self.src_r,self.src_g,self.src_b,self.src_re,self.src_nir]
        all_src =[('red',self.src_r),('green',self.src_g),('blue',self.src_b),('re', self.src_re),('nir',self.src_nir)]

        for band_name,band_src in all_src:    

            output_path = output_path_clip + f'/{band_name}/'

            if not os.path.exists(output_path):
                 os.makedirs(output_path)

            # Clip image to the shapefile geometry
            clipped_image, clipped_transform = mask(band_src, self.gdf.geometry, crop=True)

            # Specify output path for each individual band using the image name
            clipped_image_path = os.path.join(output_path, f'{band_name}-clipped.tif')

            # Create a new raster file with the same dimensions as the clipped band
            with rasterio.open(clipped_image_path, 'w', driver='GTiff', width=clipped_image.shape[2], height=clipped_image.shape[1], count=clipped_image.shape[0], dtype=clipped_image.dtype, crs=band_src.crs, transform=clipped_transform) as dst:
                dst.write(clipped_image)
            print(f'clipped image {band_name}-clipped.tif saved to {clipped_image_path}')
            
            # Print shape and CRS (sanity check)
            print(f'Shape: {clipped_image.shape}, CRS: {band_src.crs}')
            
            # loop through the selected polygons and save each clipped image separately
            for polygon_idx, polygon in enumerate(self.gdf.geometry):
                # Extract the geometry of the selected polygon
                polygon_geometry = self.gdf.iloc[polygon_idx]['geometry']
                plot_name = self.gdf.iloc[polygon_idx]['Plot_ID']   # here we cannot fix it, should be changable.

                # Clip the image to the selected polygon's geometry and specify the output CRS
                clipped_image, clipped_transform = mask(band_src, [polygon_geometry], crop=True)
                clipped_meta = band_src.meta.copy()
                clipped_meta['crs'] = {'init': f'epsg:{self.gdf.crs}'}

                # output file path for the clipped image using the band and location index
                output_image_path = os.path.join(output_path, f'{plot_name}.tif')

                # Create a new raster file with the same dimensions as the clipped image
                with rasterio.open(output_image_path, 'w', driver='GTiff', width=clipped_image.shape[2], height=clipped_image.shape[1], count=clipped_image.shape[0], dtype=clipped_image.dtype, crs=band_src.crs, transform=clipped_transform) as dst:
                    dst.write(clipped_image)

            print("Clipped images saved in the output directories for each plot.")

        return True    

    def dataset_extraction_auto(self,target_df):

        # check if there is any values in Plot_ID are same for cc_df and target_df
        common_value_exist = bool(set(self.gdf['Plot_ID']).intersection(set(target_df['Plot_ID'])))

        if common_value_exist == False:
            return

        # Create output directory
        output_path = self.output_dir +'/Excel/'

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        col_name =['NDVI', 'GNDVI', 'DVI', 'EVI', 'EVI2','ARI', 'MARI', 'CHLRE', 'CHLGR', 'SAVI', 'SAVI2', 'MSAVI', 'OSAVI', 'TSAVI', 'MSR', 'MCARI', 'MCARIDIOSAVI', 'PVI', 'SR', 'WDRI']
        column_num = len(col_name)

        plot_num = self.gdf.shape[0]
        # threshold_all = [[None] * column_num for _ in range(plot_num)]
        cc_all = [[None] * column_num for _ in range(plot_num)]
        red_all =[[None] * column_num for _ in range(plot_num)]
        green_all =[[None] * column_num for _ in range(plot_num)]
        blue_all = [[None] * column_num for _ in range(plot_num)]
        nir_all =[[None] * column_num for _ in range(plot_num)]
        rededge_all = [[None] * column_num for _ in range(plot_num)]

        # loop through the selected polygons and save each clipped image separately
        for polygon_idx, polygon in enumerate(self.gdf.geometry):
            # Extract the geometry of the selected polygon
            polygon_geometry = self.gdf.iloc[polygon_idx]['geometry']
            plot_name = self.gdf.iloc[polygon_idx]['Plot_ID']

            # Clip the image to the selected polygon's geometry and specify the output CRS
            clipped_image_r, clipped_transform_r = mask(self.src_r, [polygon_geometry], crop=True)
            clipped_image_g, clipped_transform_g = mask(self.src_g, [polygon_geometry], crop=True)
            clipped_image_b, clipped_transform_b = mask(self.src_b, [polygon_geometry], crop=True)
            clipped_image_re, clipped_transform_re = mask(self.src_re, [polygon_geometry], crop=True)
            clipped_image_nir, clipped_transform_nir = mask(self.src_nir, [polygon_geometry], crop=True)


            clipped_images=[('red',clipped_image_r),('green',clipped_image_g),('blue',clipped_image_b),('rededge',clipped_image_re),('nir',clipped_image_nir)]

            # clear out the background fill
            for name,clipped_image in clipped_images:
                mask_black = (clipped_image<=0)
                masked_img = np.where(mask_black,np.nan, clipped_image)
                if name == 'red':
                    red = masked_img[0,:,:]
                elif name =='green':
                    green = masked_img[0,:,:]
                elif name == 'blue':
                    blue = masked_img[0,:,:]
                elif name=='rededge':
                    rededge = masked_img[0,:,:]
                elif name =='nir':
                    nir = masked_img[0,:,:]

            VIs = MS2Dataset.calculate_VI(red,green,blue,rededge,nir,None)

            cc =[]
            red_mean =[]
            green_mean =[]
            blue_mean = []
            rededge_mean =[]
            nir_mean =[]

            for (title, vi) in VIs:
                vi_values = vi.flatten() # Flatten the image array to a 1D array
                vi_values = vi_values[~np.isnan(vi_values)]  # keep the non-NA vi values
                threshold_new = otsu(vi_values,False)     # calculate thresholds 
                var_tuple ={'vi':vi,'threshold':threshold_new,'red':red,'green':green, 'blue':blue,'re':rededge, 'nir':nir}
                output = MS2Dataset.calculate_canpoy_cover(var_tuple)  # calculate the vegetation ratio
                cc.append(output['cc'])
                red_mean.append(output['red'])
                green_mean.append(output['green'])
                blue_mean.append(output['blue'])
                rededge_mean.append(output['re'])
                nir_mean.append(output['nir'])


            cc_all[polygon_idx]= cc
            red_all[polygon_idx]= red_mean
            green_all[polygon_idx] = green_mean
            blue_all[polygon_idx]= blue_mean
            rededge_all[polygon_idx]= rededge_mean
            nir_all[polygon_idx]= nir_mean

        # Save all cc to a dataframe          
        cc_df = pd.DataFrame(cc_all, columns= col_name)
        cc_df.insert(0,column='Plot_ID',value=self.gdf['Plot_ID'])

        red_df = pd.DataFrame(red_all, columns = col_name)
        red_df.insert(0,column='Plot_ID',value=self.gdf['Plot_ID'])
        green_df = pd.DataFrame(green_all, columns = col_name)
        green_df.insert(0,column='Plot_ID',value=self.gdf['Plot_ID'])
        blue_df = pd.DataFrame(blue_all, columns = col_name)
        blue_df.insert(0,column='Plot_ID',value=self.gdf['Plot_ID'])
        re_df = pd.DataFrame(rededge_all, columns = col_name)
        re_df.insert(0,column='Plot_ID',value=self.gdf['Plot_ID'])
        nir_df = pd.DataFrame(blue_all, columns = col_name)
        nir_df.insert(0,column='Plot_ID',value=self.gdf['Plot_ID'])
    

        # merge the yld dataframe and cc_all to make sure they are in the same order
        merged_df_cc = pd.merge(cc_df,target_df,how='inner', left_on='Plot_ID',right_on='Plot_ID')
        merged_df_red = pd.merge(red_df,target_df,how='inner', left_on='Plot_ID',right_on='Plot_ID')
        merged_df_green = pd.merge(green_df,target_df,how='inner', left_on='Plot_ID',right_on='Plot_ID')
        merged_df_blue = pd.merge(blue_df,target_df,how='inner', left_on='Plot_ID',right_on='Plot_ID')
        merged_df_re = pd.merge(re_df,target_df,how='inner', left_on='Plot_ID',right_on='Plot_ID')
        merged_df_nir = pd.merge(nir_df,target_df,how='inner', left_on='Plot_ID',right_on='Plot_ID')


        # calculate the correlation and find max one.(It will write all of them if the max are more than one)
        target_col_name = merged_df_cc.columns[-1]
        correlations = {}
        for col in merged_df_cc.columns[1:-1]:
            correlation = merged_df_cc[col].corr(merged_df_cc[target_col_name]) # last column is target 
            correlations[col] = correlation
        print(correlations) 

        # Save the cc has highest correlation with yield to new dataframe
        max_corr_value = max(correlations.values())   # max corr value
        print(f" The max value is: {max_corr_value}")
        max_corr_columns = [col for col, correlation in correlations.items() if correlation == max_corr_value] # all columns name with max corr values
        round_max_corr = r'{:.2f}'.format(float(max_corr_value))
        print(max_corr_columns,max_corr_value)

        for col in max_corr_columns:
            df_new = merged_df_cc.iloc[:,:1]
            df_new[target_col_name] = merged_df_cc[target_col_name]
            df_new['CC'] = merged_df_cc[f'{col}']

            # max_corr_index = merged_df_cc.columns.get_loc(col)
            # print(max_corr_index)
            red = merged_df_red[f'{col}']
            green = merged_df_green[f'{col}']
            blue = merged_df_blue[f'{col}']
            rededge = merged_df_re[f'{col}']
            nir = merged_df_nir[f'{col}']

            VIs = MS2Dataset.calculate_VI(red,green,blue,rededge,nir,None)

            data_df = [('red', red),('green',green),('blue',blue),('rededge',rededge),('nir',nir)]+VIs

            # dataframe
            for colname,coldata in data_df:
                df_new[colname] = coldata

            # write to the excel
            excel_filepath = os.path.join(output_path, f'dataset_{col}_{round_max_corr}.xlsx')
            df_new.to_excel(excel_filepath, index=False)  

    # Test: select a plot to see
    def visualization_plot(self, plot_index,is_hist,selected_band_show):

        # Extract the geometry of the selected polygon
        polygon_geometry = self.gdf.iloc[plot_index]['geometry']
        plot_name = self.gdf.iloc[plot_index]['Plot_ID']

        red,green,blue,rededge,nir = self.mask_black(polygon_geometry)

        # VI(vegetation index)
        VIs = MS2Dataset.calculate_VI(red,green,blue,rededge,nir,None)
        
        if is_hist:
            fig, axs = plt.subplots(nrows=5, ncols=4, figsize=(12, 8))
            axs = np.ravel(axs)

            threshold = []
            for ax, (title, vi) in zip(axs, VIs):
                vi_values = vi.flatten() # Flatten the image array to a 1D array
                vi_values = vi_values[~np.isnan(vi_values)]
                plot_VI_hist(ax,title,vi_values,plot_name)     # Plot histogram of pixel values
            plt.suptitle('vegetation index hist')
        else:

            match selected_band_show:
                case 'red':
                    selected_band = red
                case 'green':
                    selected_band = green
                case 'blue':
                    selected_band = blue
                case 'rededge':
                    selected_band = rededge
                case 'nir':
                    selected_band = nir
                case _:
                    selected_band = nir   
                    

            threshold_all = []

            for (title, vi) in VIs:
                vi_values = vi.flatten() # Flatten the image array to a 1D array
                vi_values = vi_values[~np.isnan(vi_values)]  # keep the non-NA vi values
                threshold = otsu(vi_values,False)     # calculate thresholds 
                threshold_all.append(threshold)
            

            VIs_th = [(v[0],v[1],t) for v,t in zip(VIs, threshold_all)]

            fig, axs = plt.subplots(nrows=5, ncols=4, figsize=(12, 8))
            axs = np.ravel(axs)

            for ax, (title, vi, threshold) in zip(axs, VIs_th):
                print(vi, threshold)
                VI_selected = vi
                threshold_selected = threshold

                # Apply thresholds to create a mask
                mask_selected =  VI_selected < threshold_selected
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
                ax.imshow(selected_band, cmap='RdYlGn', aspect=1)
                ax.axis('off')
                ax.set_title(f'{title} : CC={vegetation_cover_ratio}')

            plt.suptitle(f'{plot_name}:vegetation pixels')
        plt.tight_layout()
        plt.show()    

    def dataset_extraction_manu_threshold(self,selected_vi):
        data_for_window = []
        
        # loop through the selected polygons and save each clipped image separately
        for polygon_idx, polygon in enumerate(self.gdf.geometry):

            # Extract the geometry of the selected polygon
            polygon_geometry = self.gdf.iloc[polygon_idx]['geometry']
            plot_name = self.gdf.iloc[polygon_idx]['Plot_ID']

            red,green,blue,rededge,nir = self.mask_black(polygon_geometry)
              
            vi = MS2Dataset.calculate_VI(red,green,blue,rededge,nir,selected_vi)
            vi_values = vi.flatten() # Flatten the image array to a 1D array
            vi_values = vi_values[~np.isnan(vi_values)]  # keep the non-NA vi values
            threshold = otsu(vi_values,False)     # calculate thresholds 
            list_data = [polygon_idx,plot_name,threshold,min(vi_values),max(vi_values)]
            data_for_window.append(list_data)

        return data_for_window    
     
    def dataset_extraction_manu_plot(self,selected_vi,threshold,plot_index, hist_or_cc,selected_band_show):
        # Extract the geometry of the selected polygon
        polygon_geometry = self.gdf.iloc[plot_index]['geometry']
        plot_name = self.gdf.iloc[plot_index]['Plot_ID']

        red,green,blue,rededge,nir = self.mask_black(polygon_geometry)
            
        vi = MS2Dataset.calculate_VI(red,green,blue,rededge,nir,selected_vi)

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

            match selected_band_show:
                case 'red':
                    selected_band = red
                case 'green':
                    selected_band = green
                case 'blue':
                    selected_band = blue
                case 'rededge':
                    selected_band = rededge
                case 'nir':
                    selected_band = nir
                case _:
                    selected_band = nir 

            # Apply thresholds to create a mask
            mask_selected =  VI_selected < threshold_selected
            VI_masked = np.where(mask_selected,np.nan,VI_selected) # mask the non veg pixels
            vegetation_pixels = np.sum(~np.isnan(VI_masked)) # extrat the veg pixels
            total_pixels = np.prod(VI_masked.shape) - np.sum(np.isnan(VI_selected)) # the total pixels
            vegetation_cover_ratio = vegetation_pixels / total_pixels  # canopy cover


            # Plot vegetaion pixels on original image
            pure_soil_pixel_coords = [(col, row) for row, col in zip(*np.where(~np.isnan(VI_masked)))]
            crs_coords = [(col, row) for col,row in pure_soil_pixel_coords]     # Get the CRS coordinates of the pure pixel coordinates

            # Plot the scatter plot for the pure pixel coordinates in CRS coordinates
            crs_x_coords, crs_y_coords = zip(*crs_coords)
            ax.scatter(crs_x_coords, crs_y_coords, c='red', marker='x', s=1, label='Pure veg Pixels')        

            # # Display image and Set the labels and colorbar for the image
            # extent = (self.src.bounds.left, self.src.bounds.right, self.src.bounds.bottom, self.src.bounds.top)
            ax.imshow(selected_band, cmap='RdYlGn', aspect=1)
            ax.axis('off')
            ax.set_title(f'{title} : CC={vegetation_cover_ratio}')

            plt.suptitle(f'{plot_name}:vegetation pixels with threshold {threshold}')
            plt.tight_layout()
            plt.show()

    def dataset_extraction_manu_ok(self,selected_vi, selected_threshold):
        # need to calculate cc, red_mean,green_mean,blue_mean,vi
        # Create output directory
        output_path = self.output_dir +'/Excel/'

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        pass

        cc =[]
        red_mean =[]
        green_mean =[]
        blue_mean = []
        rededge_mean =[]
        nir_mean = []

        # with rasterio.open(image_path) as src:
        # loop through the selected polygons and save each clipped image separately
        for polygon_idx, polygon in enumerate(self.gdf.geometry):

            # Extract the geometry of the selected polygon
            polygon_geometry = self.gdf.iloc[polygon_idx]['geometry']

            red,green,blue,rededge,nir = self.mask_black(polygon_geometry)

            # VI(vegetation index)    
            vi = MS2Dataset.calculate_VI(red,green,blue,rededge,nir,selected_vi)
            threshold = selected_threshold[polygon_idx]


            var_tuple ={'vi':vi,'threshold':threshold,'red':red,'green':green, 'blue':blue,'re':rededge, 'nir':nir}
            output = MS2Dataset.calculate_canpoy_cover(var_tuple)  # calculate the vegetation ratio

            cc.append(output['cc'])
            red_mean.append(output['red'])
            green_mean.append(output['green'])
            blue_mean.append(output['blue'])
            rededge_mean.append(output['re'])
            nir_mean.append(output['nir'])

        
        VIs = MS2Dataset.calculate_VI(np.array(red_mean).astype(float),np.array(green_mean).astype(float),np.array(blue_mean).astype(float),np.array(rededge_mean).astype(float),np.array(nir_mean).astype(float),None)

        data_list = [('Plot_ID',self.gdf['Plot_ID']),(f'CC_{selected_vi}',cc),('red',red_mean),('green',green_mean),('blue',blue_mean),('rededge',rededge_mean),('nir',nir_mean)]+ VIs

        data_dict ={col_name: data for col_name, data in data_list } 
 
        df = pd.DataFrame(data_dict)

        # write to the excel
        excel_filepath = os.path.join(output_path, f'dataset_{selected_vi}.xlsx')
        df.to_excel(excel_filepath, index=False)

        print('Dataset saved!!')            


    def gdf_visualization(self):
        ax = self.gdf.geometry.plot(figsize =(12,8))
        self.gdf.apply(lambda x: ax.annotate(text=x['Plot_ID'], xy=x.geometry.centroid.coords[0], ha='center'), axis=1) 

    def check_clipped(self,selected_band):

        match selected_band:
            case 'red':
                src = self.src_r
            case 'green':
                src = self.src_g
            case 'blue':
                src =self.src_b
            case 'rededge':
                src = self.src_re
            case 'nir':
                src = self.src_nir
            case _:
                src = self.src_nir

        clipped_image, clipped_transform = mask(src, self.gdf.geometry, crop=True)
        mask_black = clipped_image < 0
        image = np.where(mask_black,np.nan, clipped_image)

        # Create a Geodata-frame with the clipped geometry
        gdf_clipped = gpd.GeoDataFrame({'geometry': [g for g in self.gdf.geometry]}, crs=self.gdf.crs)


        # Plot each band image with the CRS coordinates and their band names
        fig, ax = plt.subplots(1, 1, figsize=(12,8))

        im = rasterio.plot.show(image, transform= clipped_transform, ax=ax, cmap='viridis')  # You can specify a colormap (e.g., 'viridis')
        # ax.imshow(image.transpose(1,2,0),cmap='viridis')
        gdf_clipped.boundary.plot(ax=ax, color='red')  # Add the shapefile boundaries for reference
        self.gdf.apply(lambda x: ax.annotate(text=x['Plot_ID'], xy=x.geometry.centroid.coords[0], ha='center',color ='red'), axis=1)
        # ax.set_title(f"{self.filename} in CRS Coordinates")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        plt.tight_layout()
        plt.show()       
