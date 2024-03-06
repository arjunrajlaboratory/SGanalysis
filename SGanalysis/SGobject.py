import numpy as np
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
import tifffile as tiff
from rasterio.features import shapes
from shapely.geometry import Point, shape, Polygon, box
import matplotlib.pyplot as plt
import json

class SGobject:
    def __init__(self):
        self.gdf = None
        self.points_gdf = None
        self.cell_count_df = None
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
            object_id.append(value)
        
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

    def create_cell_count_df(self, index_col='object_id', point_assignment_mode=None):
        """Creates a DataFrame counting occurrences of points within each cell and reports assignment percentage.
        Optionally sets the point_assignment_mode if provided.
        
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
        
        # Reset the index so the index_col becomes a regular column
        # pivot_table.reset_index(inplace=True)
        self.cell_count_df = pivot_table
        
    def plot_gene_scatter(self, gene1, gene2):
        """Plots a scatter plot comparing occurrences of two genes across cells.

        Parameters:
        - gene1: The name of the first gene to plot on the x-axis.
        - gene2: The name of the second gene to plot on the y-axis.
        """
        # Check for the existence and non-emptiness of cell_count_df
        if self.cell_count_df is None or self.cell_count_df.empty:
            print("cell_count_df is missing or empty. Please run the function to create it.")
            return
        
        # Check if both genes exist in the columns of cell_count_df
        if gene1 not in self.cell_count_df.columns or gene2 not in self.cell_count_df.columns:
            print(f"One or both genes ({gene1}, {gene2}) not found in cell_count_df.")
            return
        
        # Extract occurrences for each gene
        gene1_counts = self.cell_count_df.get(gene1, 0)  # Default to 0 if gene not found
        gene2_counts = self.cell_count_df.get(gene2, 0)
        
        # Plot scatter
        plt.figure(figsize=(8, 6))
        plt.scatter(gene1_counts, gene2_counts, alpha=0.6, edgecolors='w', color='blue')
        plt.title(f"Scatter Plot of {gene1} vs. {gene2}")
        plt.xlabel(gene1)
        plt.ylabel(gene2)
        plt.grid(True)
        plt.show()

    def show_gene_stats_plots(self, gene_name = None):
        """Shows plots of gene statistics for the assigned points.

        Parameters:
        - gene_name: The name of the gene to plot.
        """
        if self.cell_count_df is None:
            print("Error: Cell Count DataFrame is not loaded.")
            return
        
        if gene_name is None:
            print("Error: Please specify a gene name to plot.")
            return
    
        if gene_name in self.cell_count_df.columns:
            # Plot histogram of expression levels for the specified gene
            self.cell_count_df[gene_name].plot(kind='hist', bins=100, title=f'Gene expression distribution for {gene_name}')
            plt.show()

            # Plot the gene expression levels on the polygons
            merged_gdf = self.gdf.merge(self.cell_count_df, on='object_id', how='left')
            merged_gdf.plot(column=gene_name, cmap='Wistia', legend=True, figsize=(10, 10))
            plt.show()


            # Print the statistics for the specified gene, including max, min, mean, median
            gene_stats = self.cell_count_df[gene_name].describe()
            print(f"Statistics for gene {gene_name}:", gene_stats)

            # Print the number of objects expressing the gene
            num_expressing = (self.cell_count_df[gene_name] > 0).sum()
            print(f"Number of objects expressing {gene_name}: {num_expressing} out of {len(self.cell_count_df)} ({num_expressing / len(self.cell_count_df) * 100:.2f}%)")
        else:
            print(f"Error: Gene '{gene_name}' not found in the cell count matrix.")

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
        if self.cell_count_df.empty:
            print("Cell Count DataFrame (cell_count_df) is missing. Please run appropriate loading function.")
            return

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

    def plot_polygon_and_points(self, identifier, id_field='object_id', gene_names=None):
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
        points_within_bbox = self.assigned_points_gdf[self.assigned_points_gdf.geometry.within(expanded_bbox)]

        if gene_names is not None:
            if isinstance(gene_names, str):
                gene_names = [gene_names]
            points_within_bbox = points_within_bbox[points_within_bbox['name'].isin(gene_names)]

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
            
            # Labeling remains the same for all points
            for x, y in zip(group.geometry.x, group.geometry.y):
                ax.text(x, y, name, fontsize=8, ha='right')

        ax.set_xlim([minx - dx, maxx + dx])
        ax.set_ylim([miny - dy, maxy + dy])
        ax.set_title(f"Polygon {identifier} and Surrounding Area")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        plt.show()