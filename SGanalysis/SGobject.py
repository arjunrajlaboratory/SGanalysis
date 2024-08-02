import numpy as np
import scanpy as sc
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
import tifffile as tiff
from rasterio.features import shapes
from shapely.geometry import Point, shape, Polygon, box
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
import random
import subprocess
import os
from shapely.affinity import scale
import warnings


class SGobject:
    def __init__(self):
        self.gdf = None
        self.points_gdf = None
        self.cell_gene_table = None
        self.assigned_points_gdf = None
        self._point_assignment_mode = "Nearest"  # Use a private variable for the actual storage

    @property
    def point_assignment_mode(self):
        return self._point_assignment_mode

    @point_assignment_mode.setter
    def point_assignment_mode(self, value):
        if value in ["Nearest", "NoOverlap", "All"]:
            self._point_assignment_mode = value
        else:
            raise ValueError("point_assignment_mode must be 'Nearest', 'NoOverlap', or 'All'.")


    def mask_to_objects(self, tiff_path, tolerance=1.5, object_column_name='nucleus'):
        """Loads a TIFF image, converts it to mask polygons, and stores them in a GeoDataFrame.

        Parameters:
        - tiff_path: Path to the TIFF file.
        - tolerance: Tolerance parameter for the simplify method. Default is 1.5. Minimum is 0.
        - object_column_name: Name for the object column. Defaults to 'nucleus'.
        """
        # Ensure the tolerance is not set below 0
        tolerance = max(tolerance, 0)

        image = tiff.imread(tiff_path)
        mask_polygons = shapes(image.astype('int32'), mask=image > 0)
        
        # Preparing lists to store polygon shapes and their associated mask values
        features = []
        object_id = []
        
        for geom, value in mask_polygons:
            simplified_shape = shape(geom).simplify(tolerance, preserve_topology=True)
            features.append(simplified_shape)
            object_id.append(str(np.int32(value))) # Could probably avoid the cast to int32
        
        self.gdf = gpd.GeoDataFrame({'object_id': object_id, object_column_name: features}, geometry=object_column_name)


    def nimbus_json_to_objects(self, json_path, object_column_name='nucleus'):
        """Loads polygon data from a JSON file, intended to represent objects,
        and stores them in a GeoDataFrame with the specified object column name as the geometry column.

        Parameters:
        - json_path: Path to the JSON file containing polygon annotations.
        - object_column_name: Name for the object column. Defaults to 'nucleus'.
        """
        # Load JSON data from file
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Initialize lists to store polygon features and their identifiers
        features = []
        object_id = []
        
        # Extract polygons and their IDs
        for annotation in data['annotations']:
            if annotation['shape'] == 'polygon':
                # Extract coordinates for each polygon and create a Shapely Polygon object
                polygon = Polygon([(coord['x'], coord['y']) for coord in annotation['coordinates']])
                features.append(polygon)
                # Extract the ID and use it as object_id
                object_id.append(annotation['id'])
        
        # Create a GeoDataFrame with the polygons as the specified object column and their IDs
        self.gdf = gpd.GeoDataFrame({'object_id': object_id, object_column_name: features}, geometry=object_column_name)

    def dilate_objects(self, radius=10.0, identifier=None, output_name=None, set_geometry=True):
        """Dilates the geometries in the GeoDataFrame by a specified radius.

        Parameters:
        - radius: The radius of dilation. Default is 10.0.
        - identifier: The column name of the geometries to dilate. Uses the current geometry if None. Default is None.
        - output_name: The name for the column of dilated geometries. Defaults to appending "_dilated" to identifier.
        - set_geometry: Whether to set the new column as the active geometry column. Default is True.
        """
        if self.gdf is None:
            print("Error: GeoDataFrame is not loaded.")
            return
        
        if identifier is None:
            identifier = self.gdf.geometry.name  # Use the current geometry column name if identifier is None
        
        if output_name is None:
            output_name = f"{identifier}_dilated"  # Default output name

        # Perform the dilation
        self.gdf[output_name] = self.gdf[identifier].buffer(radius)
        
        if set_geometry:
            self.gdf = self.gdf.set_geometry(output_name)

        print(f"Dilation completed. Output column: '{output_name}'{' is now the active geometry column.' if set_geometry else '.'}")


    def save_geojson_polygons(self, file_path):
        """Saves the polygons stored in the GeoDataFrame to a GeoJSON file."""
        if self.gdf is not None:
            self.gdf.to_file(file_path, driver='GeoJSON')
    
    def load_geojson_polygons(self, file_path):
        """Loads polygons from a GeoJSON file into the GeoDataFrame."""
        self.gdf = gpd.read_file(file_path)

    def load_points(self, csv_path):
        """Loads point data from a CSV file and creates a points GeoDataFrame."""
        df = pd.read_csv(csv_path, usecols=['name', 'refid', 'x', 'y'], dtype={'refid': str})
        # Convert the 'x' and 'y' columns to a GeoSeries of Point geometries
        self.points_gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df.x, df.y)])
        # Optionally, drop the 'x' and 'y' columns if they are not needed
        # self.points_gdf.drop(columns=['x', 'y'], inplace=True)

    def create_cell_gene_table(self, index_col='object_id', point_assignment_mode=None):
        """Creates an AnnData object counting occurrences of points within each cell and reports assignment percentage.
        Optionally sets the point_assignment_mode if provided. Generally, it is better not to directly access the AnnData object.
        Rather, you can use the get_cell_gene_table_df() function to get the cell_gene_table as a DataFrame.
        
        Parameters:
        - index_col: The column name to use as the index in the pivot table. Default is 'object_id'.
        - point_assignment_mode: Optional; Specifies the mode for point assignment. Default is None.
        """
        if self.points_gdf is None or self.gdf is None:
            print("Error: Ensure both points_gdf and gdf are loaded.")
            return
        
        # Optionally set the point_assignment_mode
        if point_assignment_mode is not None:
            self.point_assignment_mode = point_assignment_mode  # This will use the setter and its validation
        else:
            self.point_assignment_mode = "Nearest"  # Default is to use the nearest object
        
        # Perform a spatial join to find points within cells using the current point_assignment_mode
        self.assigned_points_gdf = gpd.sjoin(self.points_gdf, self.gdf, how='left', predicate='within')

        # Right now, we are calculating the distance to the nearest edge for each point.
        # In principle, doing this for just the points assigned to multiple objects would be faster,
        # but in practice it doesn't seem to be much faster if at all, so figure we might as well compute all this stuff.
        # Define a function to calculate distance to the nearest polygon edge
        def distance_to_polygon_edge(point, polygon):
            # Check if polygon is None before attempting to access its boundary
            if polygon is None:
                return None
            # If polygon is not None, proceed to calculate the distance
            return point.distance(polygon.boundary)
        
        self.assigned_points_gdf['distance_to_edge'] = self.assigned_points_gdf.apply(
            lambda row: distance_to_polygon_edge(
                row.geometry, 
                # Use None if 'index_right' is nan; otherwise, get the polygon geometry
                self.gdf.geometry.loc[row['index_right']] if not pd.isna(row['index_right']) else None
            ), 
            axis=1
        )

        if self.point_assignment_mode == "Nearest":
            # Keep only the multiply-associated points that have the biggest value of "distance_to_edge"
            self.assigned_points_gdf = self.assigned_points_gdf.sort_values(by="distance_to_edge", ascending=False)
            self.assigned_points_gdf.drop_duplicates(subset=["name", "geometry"], keep="first", inplace=True)
        elif self.point_assignment_mode == "NoOverlap":
            # Remove all the points that are assigned to multiple objects
            self.assigned_points_gdf = self.assigned_points_gdf[self.assigned_points_gdf[index_col].notnull()]
        elif self.point_assignment_mode == "All":
            # No action needed, keep all the points
            pass
        else:
            print("Error: Invalid point_assignment_mode. Please choose from 'Nearest', 'NoOverlap', or 'All'.")
            return

        
        # Continue with your existing logic using self.assigned_points_gdf
        total_points = len(self.assigned_points_gdf)
        assigned_points = self.assigned_points_gdf[index_col].notnull().sum()  # Count non-null entries for the specified index column
        
        # Calculate the percentage of points assigned to a cell/polygon
        percent_assigned = (assigned_points / total_points) * 100
        
        # Print out the assignment report
        print(f"{assigned_points} of {total_points} spots ({percent_assigned:.1f}%) assigned to an object.")
        
        # Create the pivot table
        pivot_table = pd.pivot_table(self.assigned_points_gdf, 
                                    values='geometry',  # Use any column that exists
                                    index=index_col,  # Rows are based on the specified index column
                                    columns='name',  # Columns are 'name'
                                    aggfunc='count',  # Count occurrences
                                    fill_value=0)  # Fill missing values with 0


        # Create the AnnData object
        self.cell_gene_table = sc.AnnData(pivot_table)
    
    def run_proseg(self, output_path='output.csv', regenerate_proseg_polygons=False):
        """Export processed data merging points_gdf and assigned_points_gdf, then save to CSV."""
        if self.points_gdf is None or self.assigned_points_gdf is None:
            raise ValueError("Points not loaded/assigned. Please load data and assign points before exporting.")

        # Merge GeoDataFrames
        merged_gdf = self.points_gdf.merge(self.assigned_points_gdf[['object_id']], left_index=True, right_index=True, how='left')

        # Replace NaN in 'object_id' with '0'
        merged_gdf['object_id'].fillna('0', inplace=True)

        # Prepare DataFrame with specified columns and transformations
        new_df = pd.DataFrame({
            'transcript_id': merged_gdf.index,
            'gene_column': merged_gdf['name'],
            'x_column': merged_gdf['x'],
            'y_column': merged_gdf['y'],
            'cell_id': merged_gdf['object_id']
        })

        # Convert 'cell_id' to integer for comparison
        new_df['cell_id'] = new_df['cell_id'].astype(int)

        # Add 'compartment_column' based on 'cell_id'
        new_df['compartment_column'] = (new_df['cell_id'] > 0).astype(int)

        # Add a new column 'z' with the value 1 for all rows
        new_df['z'] = 1

        # Save to CSV
        new_df.to_csv(output_path, index=False)

        print(f"Data exported to {output_path}")

        geojson_path = 'cell-polygons.geojson'

        # Check if the output file exists and conditionally run proseg and gunzip
        if not os.path.exists(geojson_path) or regenerate_proseg_polygons:
            print("Running Proseg...")
            # Build the command to run proseg
            proseg_command = [
                'proseg',
                '--gene-column', 'gene_column',
                '--transcript-id-column', 'transcript_id',
                '--x-column', 'x_column',
                '--y-column', 'y_column',
                '--z-column', 'z',
                '--compartment-column', 'compartment_column',
                '--compartment-nuclear', '1',
                '--cell-id-column', 'cell_id',
                '--cell-id-unassigned', '0',
                '--ignore-z-coord',
                '--no-diffusion',
                '--diffusion-probability', '0.01',
                '--coordinate-scale', '0.107',
                '--enforce-connectivity',
                output_path
            ]

            # Execute the proseg command
            try:
                result = subprocess.run(proseg_command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print("Proseg Output:", result.stdout)
                print("Proseg Errors (if any):", result.stderr)
            except subprocess.CalledProcessError as e:
                print("Error during Proseg execution:", e)
                print(e.stdout)
                print(e.stderr)
                return

            # Command to decompress the geojson file
            if not os.path.exists(geojson_path):
                gunzip_command = ['gunzip', 'cell-polygons.geojson.gz']

                # Execute the gunzip command
                try:
                    subprocess.run(gunzip_command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    print("Decompression complete: cell-polygons.geojson.gz")
                except subprocess.CalledProcessError as e:
                    print("Error during decompression:", e)
                    print(e.stdout)
                    print(e.stderr)
        else:
            print("Proseg output already exists. Skipping Proseg and gunzip.")
        
        print("Loading cell polygons from proseg output and processing...")
        self.gdf = self._load_and_process_geometries(geojson_path)
        print("New GDF loaded and processed. Use create_cell_gene_table() to generate the cell_gene_table.")

    def _load_and_process_geometries(self, geojson_path):
        # Load GeoJSON file produced by proseg
        geo_df = gpd.read_file(geojson_path)

        # Scale geometries
        scale_factor = 1 / 0.107
        geo_df['geometry'] = geo_df['geometry'].apply(lambda geom: scale(geom, xfact=scale_factor, yfact=scale_factor, origin=(0, 0)))

        # Process geometries to simplify and select largest Polygon if MultiPolygon
        processed_geometries = []
        count_multipolygon = 0
        count_polygon = 0

        for geom in geo_df.geometry:
            if geom.geom_type == 'MultiPolygon':
                if len(geom.geoms) > 1: # if there are multiple polygons in the multipolygon
                    count_multipolygon += 1
                    largest_polygon = max(geom.geoms, key=lambda x: x.area)
                else:
                    count_polygon += 1
                    largest_polygon = geom.geoms[0]
                simplified_polygon = largest_polygon.simplify(1.43, preserve_topology=True)
                processed_geometries.append(simplified_polygon)
            else:
                count_polygon += 1
                simplified_polygon = geom.simplify(1.43, preserve_topology=True)
                processed_geometries.append(simplified_polygon)

        geo_df['geometry'] = processed_geometries  # Replace original with simplified
        geo_df.set_geometry('geometry', inplace=True)  # Ensure 'geometry' is still the active geometry column

        print(f"Total MultiPolygons processed into largest Polygons: {count_multipolygon}")
        print(f"Total Polygons simplified: {count_polygon}")

        if 'cell' in geo_df.columns:
            print("cell column exists")
            geo_df['object_id'] = geo_df['cell'].astype(str)
            geo_df.drop(columns=['cell'], inplace=True)

        # Replace the original geometry column with the processed geometries
        geo_df['cell'] = processed_geometries  # Assign processed geometries to new 'cell' column


        # Set 'cell' as the active geometry column
        geo_df.set_geometry('cell', inplace=True)
        geo_df.drop(columns=['geometry'], inplace=True)

        return geo_df


    def add_filter(self, filter_mask, filter_column="default_filter"):
        """Adds a filter as a column to the obs DataFrame of the AnnData object.

        Parameters:
        - filter_mask: Boolean mask to apply as a filter.
        - filter_column: Name of the column in obs where the filter will be stored (default is 'default_filter').
        """
        if self.cell_gene_table is None:
            print("Error: AnnData object not initialized.")
            return

        # Ensure the filter mask has the correct length
        if len(filter_mask) != self.cell_gene_table.n_obs:
            print("Error: Filter mask length must match the number of observations in the AnnData object.")
            return

        # Add the filter mask to the obs DataFrame
        self.cell_gene_table.obs[filter_column] = filter_mask
        print(f"Filter added to obs DataFrame under column '{filter_column}'.")

    def remove_filters(self, filter_name=None):
        """Removes specified filters or all filters from the obs DataFrame of the AnnData object.

        Parameters:
        - filter_name: The name of the filter to remove. If None, all filters are removed (default is None).
        """
        if self.cell_gene_table is None:
            print("Error: AnnData object not initialized.")
            return

        if filter_name is None:
            # Remove all filters; assumes all Boolean columns in obs are filters.
            # This can be customized to target specific columns if needed.
            filter_columns = [col for col in self.cell_gene_table.obs.columns if self.cell_gene_table.obs[col].dtype == 'bool']
            self.cell_gene_table.obs.drop(columns=filter_columns, inplace=True)
            print("All filters removed from obs DataFrame.")
        else:
            # Remove a specific filter
            if filter_name in self.cell_gene_table.obs.columns:
                self.cell_gene_table.obs.drop(columns=[filter_name], inplace=True)
                print(f"Filter '{filter_name}' removed from obs DataFrame.")
            else:
                print(f"Error: Filter '{filter_name}' not found in obs DataFrame.")



    # Make a function to return the cell_gene_table as a DataFrame
    def get_cell_gene_table_df(self, **kwargs):
        use_filter = kwargs.get('use_filter', True)  # Default to True if use_filter is not provided
        """Returns the cell_gene_table as a DataFrame, optionally applying a filter.

        Parameters:
        - use_filter: If True (default), applies the 'default_filter'. If False, no filter is applied.
                    If a string, it uses that as the column name for filtering.
        """
        if self.cell_gene_table is None:
            print("Error: Cell Count DataFrame is not loaded.")
            return

        df = self.cell_gene_table.to_df()

        # Determine the filter to apply
        filter_column = "default_filter"
        if isinstance(use_filter, str):
            filter_column = use_filter

        if use_filter:
            if filter_column in self.cell_gene_table.obs:
                df = df[self.cell_gene_table.obs[filter_column]]
                num_filtered = df.shape[0]
                num_total = self.cell_gene_table.n_obs
                print(f"*** NOTE: {filter_column} being applied, using {num_filtered} of {num_total} total objects ***")
            else:
                if not filter_column in self.cell_gene_table.obs:
                    print("No filtering available or applied.")
                else:
                    print(f"Filter column '{filter_column}' not applied. Returning unfiltered data.")
        return df

        
    def plot_gene_scatter(self, gene1, gene2, **kwargs):
        """Plots a scatter plot comparing occurrences of two genes across cells.

        Parameters:
        - gene1: The name of the first gene to plot on the x-axis.
        - gene2: The name of the second gene to plot on the y-axis.
        """
        # Check for the existence of cell_gene_table
        if self.cell_gene_table is None:
            print("cell_gene_table is missing. Please run the function to create it.")
            return
        
        # Lot easier to work with a dataframe than an AnnData object
        cell_gene_table_df = self.get_cell_gene_table_df(**kwargs)
        
        # Check if both genes exist in the columns of cell_gene_table
        if gene1 not in cell_gene_table_df.columns or gene2 not in cell_gene_table_df.columns:
            print(f"One or both genes ({gene1}, {gene2}) not found in cell_gene_table.")
            return
        
        # Extract occurrences for each gene
        gene1_counts = cell_gene_table_df.get(gene1, 0)  # Default to 0 if gene not found
        gene2_counts = cell_gene_table_df.get(gene2, 0)
        
        # Plot scatter
        plt.figure(figsize=(8, 6))
        plt.scatter(gene1_counts, gene2_counts, alpha=0.6, edgecolors='w', color='blue')
        plt.title(f"Scatter Plot of {gene1} vs. {gene2}")
        plt.xlabel(gene1)
        plt.ylabel(gene2)
        plt.grid(True)
        plt.show()

    def show_gene_stats_plots(self, gene_name = None, **kwargs):
        """Shows plots of gene statistics for the assigned points.

        Parameters:
        - gene_name: The name of the gene to plot.
        """
        if self.cell_gene_table is None:
            print("Error: Cell Count DataFrame is not loaded.")
            return
        
        if gene_name is None:
            print("Error: Please specify a gene name to plot.")
            return
        
        # Lot easier to work with a dataframe than an AnnData object
        cell_gene_table_df = self.get_cell_gene_table_df(**kwargs)

        if gene_name in cell_gene_table_df.columns:
            # Plot histogram of expression levels for the specified gene
            cell_gene_table_df[gene_name].plot(kind='hist', bins=100, title=f'Gene expression distribution for {gene_name}')
            plt.show()

            # Plot the gene expression levels on the polygons
            merged_gdf = self.gdf.merge(cell_gene_table_df, on='object_id', how='left')
            fig, ax = plt.subplots(figsize=(10, 10))
            merged_gdf.plot(column=gene_name, cmap='Wistia', legend=True, ax=ax)
            ax.set_aspect('equal')
            plt.show()


            # Print the statistics for the specified gene, including max, min, mean, median
            gene_stats = cell_gene_table_df[gene_name].describe()
            print(f"Statistics for gene {gene_name}:", gene_stats)

            # Print the number of objects expressing the gene
            num_expressing = (cell_gene_table_df[gene_name] > 0).sum()
            print(f"Number of objects expressing {gene_name}: {num_expressing} out of {len(cell_gene_table_df)} ({num_expressing / len(cell_gene_table_df) * 100:.2f}%)")
        else:
            print(f"Error: Gene '{gene_name}' not found in the cell count matrix.")

    def get_observation_list(self):
        """Returns a list of the observation names in the cell_gene_table."""
        if self.cell_gene_table is None:
            print("Error: Cell Count DataFrame is not loaded.")
            return
        return self.cell_gene_table.obs.columns.tolist()
    
    def plot_observation_variable(self, observation_name):

        obs_df = self.cell_gene_table.obs

        joined_gdf = self.gdf.join(obs_df, on="object_id")
        fig, ax = plt.subplots(figsize=(10, 10))
        joined_gdf.plot(column=observation_name, legend=True, ax=ax)
        ax.set_aspect('equal')
        plt.show()

    def generate_statistics_and_histograms(self):
        # Check for the existence of required properties
        if self.gdf.empty:
            print("GeoDataFrame (gdf) is missing. Please run appropriate loading function.")
            return
        if self.points_gdf.empty:
            print("Points GeoDataFrame (points_gdf) is missing. Please run appropriate loading function.")
            return
        if self.assigned_points_gdf.empty:
            print("Assigned Points GeoDataFrame (assigned_points_gdf) is missing. Please run appropriate loading function.")
            return
        if self.cell_gene_table is None:
            print("Cell Count DataFrame (cell_gene_table) is missing. Please run appropriate loading function.")
            return
        
        # Lot easier to work with a dataframe than an AnnData object
        cell_gene_table_df = self.get_cell_gene_table_df(use_filter=False)

        # Basic Statistics
        num_objects = len(self.gdf)
        num_points = len(self.points_gdf)
        avg_points_per_object = num_points / num_objects if num_objects else 0
        assigned_points = self.assigned_points_gdf.dropna(subset=['index_right'])  # Assuming 'index_right' is used for join
        percent_assigned = 100 * len(assigned_points) / num_points if num_points else 0
        avg_assigned_points_per_object = len(assigned_points) / num_objects if num_objects else 0
        
        # Unique genes detected per object
        genes_per_object = assigned_points.groupby('index_right')['name'].nunique()
        avg_genes_per_object = genes_per_object.mean()

        # Total unique genes detected and total unique genes assigned to objects
        total_genes = len(self.points_gdf['name'].unique())
        total_assigned_genes = len(self.assigned_points_gdf['name'].unique())
        
        print(f"Number of objects: {num_objects}")
        print(f"Number of points: {num_points}")
        print(f"Average number of points per object: {avg_points_per_object:.2f}")
        print(f"Percentage of points assigned to an object: {percent_assigned:.2f}%")
        print(f"Average number of assigned points per object: {avg_assigned_points_per_object:.2f}")
        print(f"Average number of unique genes detected per object: {avg_genes_per_object:.2f}")
        print(f"Total number of genes detected: {total_genes}")
        print(f"Total number of genes detected (assigned to objects): {total_assigned_genes}")


        # Histograms
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.hist(genes_per_object, bins=np.arange(genes_per_object.min(), genes_per_object.max() + 1), color='skyblue', edgecolor='black')
        plt.title('Number of Genes Detected per Object')
        plt.xlabel('Number of Genes')
        plt.ylabel('Frequency')

        plt.subplot(1, 2, 2)
        points_per_object = assigned_points.groupby('index_right').size()
        plt.hist(points_per_object, bins=np.arange(points_per_object.min(), points_per_object.max() + 1), color='lightgreen', edgecolor='black')
        plt.title('Number of Points Detected per Object')
        plt.xlabel('Number of Points')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()


    def plot_polygon_by_id(self, identifier, id_field='object_id'):
        """Plots a polygon specified by its identifier.

        Parameters:
        - identifier: The identifier of the polygon to plot.
        - id_field: The column name that contains the identifier. Default is 'object_id'.
        """
        if self.gdf is None:
            print("Error: GeoDataFrame is not loaded.")
            return
        
        # Filter the GeoDataFrame for the polygon with the specified identifier
        polygon_gdf = self.gdf[self.gdf[id_field] == identifier]
        
        # Check if the polygon was found
        if polygon_gdf.empty:
            print(f"No polygon found with {id_field} == {identifier}")
            return
        
        # Plot the polygon
        fig, ax = plt.subplots()
        polygon_gdf.plot(ax=ax, color='blue', edgecolor='black')
        
        # Enhancements for better visualization
        ax.set_title(f"Polygon with {id_field} == {identifier}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.show()

    def plot_polygon_and_points(self, identifier, id_field='object_id', gene_names=None,annotate=True):
        if self.gdf is None or self.assigned_points_gdf is None:
            print("Error: Ensure both gdf and assigned_points_gdf are loaded.")
            return

        polygon_gdf = self.gdf[self.gdf[id_field] == identifier]

        if polygon_gdf.empty:
            print(f"No polygon found with {id_field} == {identifier}")
            return

        first_polygon_geometry = polygon_gdf.geometry.iloc[0]

        minx, miny, maxx, maxy = first_polygon_geometry.bounds
        dx = (maxx - minx) * 0.5
        dy = (maxy - miny) * 0.5
        expanded_bbox = box(minx - dx, miny - dy, maxx + dx, maxy + dy)

        other_polygons = self.gdf[self.gdf.geometry.intersects(expanded_bbox) & (self.gdf[id_field] != identifier)]
        
        
        if gene_names is not None:
            if isinstance(gene_names, str):
                gene_names = [gene_names]
            # first pick the subset of points corresponding to the gene names
            points_within_bbox = self.assigned_points_gdf[self.assigned_points_gdf['name'].isin(gene_names)]
            # then filter to those within the expanded bbox
            points_within_bbox = points_within_bbox[points_within_bbox.geometry.within(expanded_bbox)]
        else:
            # if no gene names are passed, get all the ones in the expanded bbox
            points_within_bbox = self.assigned_points_gdf[self.assigned_points_gdf.geometry.within(expanded_bbox)]

        fig, ax = plt.subplots()
        polygon_gdf.boundary.plot(ax=ax, color='red', linewidth=2)
        other_polygons.boundary.plot(ax=ax, color='black', linewidth=1)

        # Generate a unique color for each name
        unique_names = points_within_bbox['name'].unique()
        color_map = {name: plt.cm.tab20(i % 20) for i, name in enumerate(unique_names)}

        # Plot points, label them, and use consistent colors for names
        for name, group in points_within_bbox.groupby('name'):
            interior_points = group[group[id_field] == identifier]
            exterior_points = group[group[id_field] != identifier]
            
            # Plot interior points with 'o' marker style
            if not interior_points.empty:
                ax.scatter(interior_points.geometry.x, interior_points.geometry.y, marker='o', s=50, edgecolor='black', color=color_map[name])
            
            # Plot exterior points with 'x' marker style
            if not exterior_points.empty:
                ax.scatter(exterior_points.geometry.x, exterior_points.geometry.y, marker='x', s=50, color=color_map[name])
            
            if annotate:
                # Labeling remains the same for all points
                for x, y in zip(group.geometry.x, group.geometry.y):
                    ax.text(x, y, name, fontsize=8, ha='right')

        ax.set_xlim([minx - dx, maxx + dx])
        ax.set_ylim([miny - dy, maxy + dy])
        ax.set_title(f"Polygon {identifier} and Surrounding Area")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        plt.show()

    def l_metric(self, gene1, gene2, plot_hexbin=False, plot_gene_counts=False, plot_cumulative_sum=False, **kwargs):
        cell_count_df = self.get_cell_gene_table_df(**kwargs)
        gene1_counts = cell_count_df.get(gene1)
        gene2_counts = cell_count_df.get(gene2)

        if plot_hexbin:
            mask = (gene1_counts > 0) | (gene2_counts > 0)
            gene1_counts_nozeros = gene1_counts[mask]
            gene2_counts_nozeros = gene2_counts[mask]
            plt.figure(figsize=(8, 6))
            plt.hexbin(gene1_counts_nozeros, gene2_counts_nozeros, gridsize=50, cmap='Reds', mincnt=1)
            plt.colorbar(label='Number of Cells')
            plt.title(f"Hexbin Plot of {gene1} vs. {gene2}")
            plt.xlabel(gene1)
            plt.ylabel(gene2)
            plt.show()

        df_genes = cell_count_df[[gene1, gene2]].copy()
        df_genes[f"random_{gene2}"] = np.random.permutation(df_genes[gene2].values)
        df_sorted = df_genes.sort_values(by=gene1, ascending=False)
        df_sorted.reset_index(drop=True, inplace=True)

        if plot_gene_counts:
            plt.figure(figsize=(10, 6))
            x_values = df_sorted.index
            plt.scatter(x_values, df_sorted[f"random_{gene2}"], color='green', label="Random", s=10, alpha=0.5)
            plt.scatter(x_values, df_sorted[gene1], color='blue', label=gene1, s=10, alpha=0.5)
            plt.scatter(x_values, df_sorted[gene2], color='red', label=gene2, s=10, alpha=0.5)
            plt.title('Gene Counts Comparison')
            plt.xlabel('Index')
            plt.ylabel('Counts')
            plt.legend()
            plt.show()

        df_sorted['cumsum_' + gene1] = df_sorted[gene1].cumsum()
        df_sorted['cumsum_' + gene2] = df_sorted[gene2].cumsum()
        df_sorted['cumsum_random_' + gene2] = df_sorted[f"random_{gene2}"].cumsum()

        if plot_cumulative_sum:
            plt.figure(figsize=(10, 6))
            plt.plot(df_sorted['cumsum_' + gene1], df_sorted['cumsum_' + gene2], color='red', label=f'{gene1} vs. {gene2}')
            plt.plot(df_sorted['cumsum_' + gene1], df_sorted['cumsum_random_' + gene2], color='green', label=f'{gene1} vs. Random {gene2}')
            plt.title(f'Cumulative Sum of {gene1} vs. {gene2}')
            plt.xlabel(f'Cumulative Sum of {gene1}')
            plt.ylabel(f'Cumulative Sum of {gene2}')
            plt.grid(True)
            plt.legend()
            plt.show()

        area_difference = 0
        total_area = 0
        for i in range(len(df_sorted) - 1):
            deltaX = df_sorted['cumsum_' + gene1].iloc[i+1] - df_sorted['cumsum_' + gene1].iloc[i]
            deltaY = ((df_sorted['cumsum_' + gene2].iloc[i+1] - df_sorted['cumsum_random_' + gene2].iloc[i+1]) + (df_sorted['cumsum_' + gene2].iloc[i] - df_sorted['cumsum_random_' + gene2].iloc[i])) / 2
            area_difference += deltaX * deltaY
            total_area += deltaX * (df_sorted['cumsum_' + gene2].iloc[i] + df_sorted['cumsum_' + gene2].iloc[i+1])/2

        area_difference_normalized = area_difference / (df_sorted['cumsum_' + gene1].iloc[-1] * df_sorted['cumsum_' + gene2].iloc[-1])
        total_area_normalized = total_area / (df_sorted['cumsum_' + gene1].iloc[-1] * df_sorted['cumsum_' + gene2].iloc[-1])
        return area_difference_normalized, total_area_normalized



    def export_to_nimbus_json(self, output_file_path, object_name=None, gene_names=None, datasetId="unknown", 
                          time=0, xy=0, z=0, object_channel=3, point_channel=0, 
                          randomize_spot_colors=True, include_connections=False, 
                          obs_variable=None):
        if not object_name:
            object_name = self.gdf.geometry.name

        if gene_names is not None:
            if isinstance(gene_names, str):
                gene_names = [gene_names]  # Convert single gene name to list

        annotations = []
        connections = []
        annotation_properties = []
        annotation_property_values = {}
        
        # Check if obs_variable exists in the obs DataFrame
        if obs_variable and obs_variable not in self.cell_gene_table.obs.columns:
            raise ValueError(f"The specified obs_variable '{obs_variable}' does not exist in the obs DataFrame.")

        # Determine if the obs_variable is categorical or continuous
        is_categorical = False
        color_mapping = {}
        if obs_variable:
            if self.cell_gene_table.obs[obs_variable].dtype.name in ['object', 'category']:
                is_categorical = True
                unique_categories = self.cell_gene_table.obs[obs_variable].unique()
                color_mapping = {cat: "#{:06x}".format(random.randint(0, 0xFFFFFF)) for cat in unique_categories}
            else:
                # For continuous variables, we'll use a colormap
                min_val = self.cell_gene_table.obs[obs_variable].min()
                max_val = self.cell_gene_table.obs[obs_variable].max()
                norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
                cmap = plt.get_cmap('viridis')

        # Add annotation property for the obs_variable
        if obs_variable:
            property_id = f"property_{obs_variable}"
            annotation_properties.append({
                "id": property_id,
                "name": f"{obs_variable} {'Category' if is_categorical else 'Value'}",
                "image": "No image",
                "tags": {
                    "exclusive": False,
                    "tags": [object_name]
                },
                "shape": "polygon",
                "workerInterface": {}
            })

        # Export polygons
        skipped_objects = 0
        for index, row in self.gdf.iterrows():
            polygon = {
                "tags": [object_name],
                "shape": "polygon",
                "channel": object_channel,
                "location": {"Time": time, "XY": xy, "Z": z},
                "coordinates": [{"x": coord[0], "y": coord[1]} for coord in list(row[object_name].exterior.coords)],
                "id": str(row['object_id']),
                "datasetId": datasetId
            }
            
            if obs_variable:
                try:
                    obs_value = self.cell_gene_table.obs.loc[row['object_id'], obs_variable]
                    if is_categorical:
                        polygon['color'] = color_mapping[obs_value]
                        polygon['tags'].append(str(obs_value))
                    else:
                        polygon['color'] = mcolors.to_hex(cmap(norm(obs_value)))
                    
                    # Add property value
                    annotation_property_values[row['object_id']] = {
                        property_id: str(obs_value)
                    }
                except KeyError:
                    skipped_objects += 1
                    warnings.warn(f"Object ID {row['object_id']} not found in cell_gene_table.obs. Skipping obs_variable for this object.")
            
            annotations.append(polygon)

        # Export points
        if gene_names:
            filtered_points_gdf = self.points_gdf[self.points_gdf['name'].isin(gene_names)]
        else:
            filtered_points_gdf = self.points_gdf

        if len(filtered_points_gdf) > 500000:
            raise ValueError("There are more than 500,000 points. Please specify gene_names to filter.")

        point_id_start = len(self.gdf) + 1
        point_id_mapping = {}  # To store mapping between point index and its new ID

        for index, row in filtered_points_gdf.iterrows():
            point_id = str(point_id_start + index)
            point_id_mapping[index] = point_id
            point = {
                "tags": [row['name']],
                "shape": "point",
                "channel": point_channel,
                "location": {"Time": time, "XY": xy, "Z": z},
                "coordinates": [{"x": row['geometry'].x, "y": row['geometry'].y, "z": 0}],
                "id": point_id,
                "datasetId": datasetId
            }
            # Add color only if randomization is enabled
            if randomize_spot_colors:
                point['color'] = color_mapping.get(row['name'], "#000000")  # Default color if not found

            annotations.append(point)

        # Add connections if include_connections is True
        if include_connections:
            connection_id_start = len(annotations) + 1
            for index, row in self.assigned_points_gdf.iterrows():
                if pd.notnull(row['object_id']):  # Check if the point is assigned to a nucleus
                    connection = {
                        "label": "(Connection) Point to Nucleus",
                        "tags": ["point_to_nucleus_connection"],
                        "id": str(connection_id_start + index),
                        "parentId": str(row['object_id']),  # The nucleus ID
                        "childId": point_id_mapping.get(index, None),  # The point ID
                        "datasetId": datasetId
                    }
                    if connection["childId"] is not None:  # Only add connection if the point was included
                        connections.append(connection)

        # Prepare the output data
        output_data = {
            "annotations": annotations,
            "annotationConnections": connections,
            "annotationProperties": annotation_properties,
            "annotationPropertyValues": annotation_property_values
        }

        with open(output_file_path, 'w') as outfile:
            json.dump(output_data, outfile, indent=2)

        print(f"Exported to JSON: {output_file_path}")
        print(f"Total annotations: {len(annotations)}")
        print(f"Total connections: {len(connections)}")
        if obs_variable:
            print(f"Added coloring and properties based on '{obs_variable}'")
            if skipped_objects > 0:
                print(f"Warning: {skipped_objects} objects were skipped due to missing data in cell_gene_table.obs")