"""
"""

from pathlib import Path
import shutil
import pandas as pd


class ProjectFilesAndPaths:
    def __init__(self, root_name: str ):
        self.root_name = Path(root_name)

        # example root_name: "DJI_0145-png_1.00_1600_none/gsplat/0/point_cloud"

        #raise Exception( Path.cwd() )
        self.local_path = "../.."
        self.web_path = ".."
        # video stuff
        self.video_name = Path(f"{str(self.root_name).split("-",1)[0] }-thumb.MP4")
        self.video_local_source = Path(f"{self.local_path}/projects/thumbvids") / self.video_name
        self.video_web_url = Path(f"{self.web_path}/data/thumbvids") / self.video_name
        self.video_local_dest = Path(f"{self.local_path}/docs") / Path( f"./data/thumbvids" ) / self.video_name

        #raise Exception( self.video_local_dest )
        
        self.video_local_dest.parent.mkdir(parents=True,exist_ok=True)
        try:
            x = shutil.copyfile( str(self.video_local_source),str(self.video_local_dest) ) 
        except (FileNotFoundError, PermissionError, shutil.SameFileError) as e:
            print(f"Error copying file: {e}")
            x = None
    
        # sparse point cloud PLY files
        self.sparse_local_source  = Path(f"{self.local_path}/projects/{self.root_name.parts[0]}/colmap/sparse/{self.root_name.parts[2]}.ply")
        self.sparse_web_url = Path(f"{self.web_path}/data") / self.root_name.parts[0] / self.sparse_local_source.name
        self.sparse_local_dest = Path(f"{self.local_path}/docs/data") / self.root_name.parts[0] / self.sparse_local_source.name
        
        #raise Exception( self.sparse_local_dest )

        self.sparse_local_dest.parent.mkdir(parents=True,exist_ok=True)
        try:
            x = shutil.copyfile( str(self.sparse_local_source), str(self.sparse_local_dest ))
        except (FileNotFoundError, PermissionError, shutil.SameFileError) as e:
            print(f"Error copying file: {e}")
            x = None

        # splat file
        self.splat_local_source  = Path(f"{self.local_path}/projects") / self.root_name / Path("iteration_30000/point_cloud.splat")
        self.splat_web_url = Path(f"{self.web_path}/data") / self.root_name.parts[0] / Path(f"{self.root_name.parts[2]}-{self.splat_local_source.name}")
        self.splat_local_dest = Path(f"{self.local_path}/docs/data") / self.root_name.parts[0] / Path(f"{self.root_name.parts[2]}-{self.splat_local_source.name}")
        try:
            x = shutil.copyfile( str( self.splat_local_source), str(self.splat_local_dest) )
        except (FileNotFoundError, PermissionError, shutil.SameFileError) as e:
            print(f"Error copying file: {e}")
            x = None

        # texture file
        self.texture_local_source  = Path(f"{self.local_path}/projects") / self.root_name.parts[0] / "mvs/0/scene_dense_mesh_texture.glb" ;
        self.texture_web_url = Path(f"{self.web_path}/data") / self.root_name.parts[0] / Path(f"{self.root_name.parts[2]}-{self.texture_local_source.name}")
        self.texture_local_dest = Path(f"{self.local_path}/docs/data") / self.root_name.parts[0] / Path(f"{self.root_name.parts[2]}-{self.texture_local_source.name}")
        try:
            x = shutil.copyfile( str( self.texture_local_source), str(self.texture_local_dest) )
        except (FileNotFoundError, PermissionError, shutil.SameFileError) as e:
            print(f"Error copying file: {e}")
            x = None

            
    def __repr__(self):
        return f"<ProjectFilesAndPaths {self.root_name} at {self.project_folder}>"
    
    
    def parse_root_to_table(self) -> pd.DataFrame:
        try:
            # Split video from the rest
            front, rest = str(self.root_name).split("/",1)
            video_part, rest = front.split('-', 1)
            # Split the rest on underscore
            fmt, fps, maxwidth, filter = rest.split('_')

            data = {
                'Field': ['Video', 'Format', 'Filter', 'FPS', 'Max Width'],
                'Value': [video_part, fmt, filter, fps, maxwidth]
            }

            return pd.DataFrame(data)

        except ValueError:
            raise ValueError("Input string is not in expected format: 'Video-Format_Filter_FPS_MaxWidth'")

    def parse_root_to_html(self) -> str:
        df = self.parse_root_to_table()
        html_table = df.to_html(index=False, classes="stats-table", border=0)
        return html_table


