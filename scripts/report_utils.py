

import shutil
import pandas as pd
from pathlib import Path
from loguru import logger
import json

def build_folder_tree_with_files(root_path):
    """
    Builds a nested dictionary representing the folder structure,
    including files listed under '__files__' in each directory.
    """
    root = Path(root_path)
    if not root.is_dir():
        raise ValueError(f"{root_path} is not a directory")

    def walk(directory):
        tree = {}
        files = []
        for item in directory.iterdir():
            if item.is_dir():
                tree[item.name] = walk(item)
            elif item.is_file():
                files.append(item.name)
        if files:
            tree['__files__'] = files
        return tree

    return {item.name: walk(item) for item in root.iterdir() if item.is_dir()}


def count_image_files(folder_path):
    """
    Counts the number of PNG and JPG files in the given folder (case-insensitive).
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"{folder_path} is not a directory")

    count = 0
    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
            count += 1
    return count


def write_index_qmd(tree, destination_folder, data_folder):
    """
    Given the 'tree' dictionary (from build_folder_tree_with_files) and a destination folder,
    writes an index.qmd file listing all projects with links to their individual QMD files.
    """
    dest = Path(destination_folder)
    dest.mkdir(parents=True, exist_ok=True)

    #logger.info( json.dumps(tree,indent=2) )

    lines = []
    lines.append("---")
    lines.append("title: \"\"")
    lines.append("---\n")

    lines.append("# Listing of pipelines\n")

    for project in sorted(tree.keys()):
        #if not project.startswith("DJI"):
        #    continue
        
        #logger.info( project )
        #video, info = project.split("-",1)

        params = {}
        params["Project"] = project
        params["Image Count"] = count_image_files( Path("projects")/project/"images" )
        df = pd.DataFrame(list(params.items()), columns=["Images attribute", "Value"])
#        thumb_video_local_file = (Path(data_folder) / "thumbvids" / f"{video}-thumb.MP4")
#        thumb_video_web_url = thumb_video_local_file.relative_to( "docs" )
        
        #logger.info( json.dumps(tree[project],indent=2) )
        
        if "colmap" in tree[project]:
            colmap_models = [ id for id in sorted(tree[project]["colmap"].keys()) if id not in ["__files__"] ]
            logger.info( colmap_models )
            colmap_df = pd.DataFrame( colmap_models,columns=["Model ID"])
            colmap_table = colmap_df.to_markdown(index=False,colalign=("left",))
        else:
            colmap_table = "<table><tr><td>No colmap models</td></tr></table>"

        if "gsplat" in tree[project]:
            gsplat_models = [ id for id in sorted(tree[project]["gsplat"].keys()) if id not in ["files"] ]
            gsplat_df = pd.DataFrame(gsplat_models, columns=["Model ID"])
            gsplat_table = gsplat_df.to_markdown(index=False,colalign=("left",))
        else:
            gsplat_table = "No gsplat models"

        if "mvs" in tree[project]:
            mvs_models = [ id for id in sorted(tree[project]["mvs"].keys()) if id not in ["files"] ]
            mvs_df = pd.DataFrame(mvs_models, columns=["Model ID"])
            mvs_table = mvs_df.to_markdown(index=False,colalign=("left",))
        else:
            mvs_table = "No mvs models"

        
        lines.append(f"## [{project}]({project}.html)")
        lines.append(f"""
:::: {{.columns}}
::: {{.column width=50%}}
### Images from video

{df.to_markdown(index=False)}

:::
::: {{.column width=50% .threejs-container}}
{project}
:::
::::

:::: {{.columns}}
::: {{.column width=30%}}
### COLMAP Models

{colmap_table}

:::
::: {{.column width=5%}}
<p>&nbsp;</p>
:::
::: {{.column width=30%}}
### GSPLAT Models

{gsplat_table}

:::
::: {{.column width=5%}}
<p>&nbsp;</p>
:::
::: {{.column width=30%}}
### MVS Models

{mvs_table}

:::
::::


""")

    index_file = dest / "index.qmd"
    index_file.write_text("\n".join(lines))
    print(f"✅ Wrote: {index_file}")


def write_project_qmd(project, tree, destination_folder, data_folder):
    """
    Given a project name, the full tree, and a destination folder,
    writes a QMD file for that project.
    """
    dest = Path(destination_folder)
    dest.mkdir(parents=True, exist_ok=True)

    contents = tree[project]

    lines = []
    lines.append(f"---\ntitle: \"\"\n---\n")
    lines.append(f"# Pipeline: {project}\n")
    

    if "mvs" in contents:
       
        models = sorted(contents["mvs"].keys())
       
        for model_id in models:
 #           if model_id not in ["files"]:
 #               continue
 
            source_file = Path("projects")/ project / "mvs" / model_id / "scene_dense_mesh_texture.glb"
            if not source_file.exists():
                continue
 
            destination_file = Path("docs") / "data" / project / "mvs" / model_id / f"scene_dense_mesh_texture.glb"
            destination_file.parent.mkdir(parents=True, exist_ok=True)

            # Copy file
            if source_file.is_file():
                shutil.copy2(source_file, destination_file)
                source_file = Path("projects")/ project / "mvs" / model_id / "scene_dense_mesh_texture_0.png"            
                destination_file = Path("docs") / "data" / project / "mvs" / model_id / f"scene_dense_mesh_texture_0.png"
                shutil.copy2(source_file, destination_file)
            else:
                continue
 
            lines.append(f"""
<div class="splat-block">
<h3>Textured GLB: Model {model_id}</h3>
<div id="glb_viewer_{project}_{model_id}" class="threejs-container"></div>
</div>
""")
            
        lines.append(f"""

<script type="module">
import {{ loadGLBViewer }} from "../js/glb_viewer.js";
""")
        for model_id in models:
#            if model_id not in ["files"]:
#                continue
            source_file = Path("projects")/ project / "mvs" / model_id / "scene_dense_mesh_texture_embedded.glb"
            if not source_file.exists():
                continue
 

            lines.append(f"loadGLBViewer(\"glb_viewer_{project}_{model_id}\"," 
                         + f"\"../data/{project}/mvs/{model_id}/scene_dense_mesh_texture_embedded.glb\","
                         + f"\"../data/{project}/mvs/{model_id}/scene_dense_mesh_texture_0.png\",'wireframe');");

        lines.append(f"</script>")
        lines.append("")

    if "gsplat" in contents:

        models = sorted(contents["gsplat"].keys())

        for model_id in models:
            point_cloud_dir = Path("projects") / project / "gsplat" / model_id / "point_cloud"
            if not point_cloud_dir.exists():
                continue


            iteration_dirs = []
            for iter_dir in point_cloud_dir.iterdir():
                if iter_dir.is_dir() and iter_dir.name.startswith("iteration_"):
                    try:
                        num = int(iter_dir.name.split("_")[1])
                        iteration_dirs.append((num, iter_dir))
                    except (IndexError, ValueError):
                        continue  # skip malformed names
                        # Sort descending by the number
            iteration_dirs.sort(reverse=True)

            # Look for iteration_* directories
            for num, iter_dir in iteration_dirs:
                splat_file = iter_dir / "point_cloud.splat"
                if not splat_file.is_file():
                    continue

                # Build a unique destination filename
                dest_name = f"{model_id}-{iter_dir.name}-gsplat-point_cloud.splat"
                destination_file = Path("docs") / "data" / project / dest_name
                destination_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(splat_file, destination_file)

                # Append to lines with unique id
                lines.append(f"""
<div class="splat-block">
<h3>Gaussian Splat: Model {model_id} ({iter_dir.name})</h3>
<div class="splat-wrapper">
<canvas id="splat_viewer_{project}_{model_id}_{iter_dir.name}" class="splat-canvas"></canvas>
</div>
</div>
""")
            
        lines.append(f"""

<script type="module">
import {{ loadSplatViewer }} from "../js/splat_viewer.js";
""")

        for model_id in models:
            point_cloud_dir = Path("projects") / project / "gsplat" / model_id / "point_cloud"
            if not point_cloud_dir.exists():
                continue

            # Look for iteration_* directories
            for iter_dir in point_cloud_dir.iterdir():
                if not iter_dir.is_dir():
                    continue
                if not iter_dir.name.startswith("iteration_"):
                    continue
                lines.append(f"loadSplatViewer(\"splat_viewer_{project}_{model_id}_{iter_dir.name}\", \"../data/{project}/{model_id}-{iter_dir.name}-gsplat-point_cloud.splat\");");

        lines.append(f"</script>")
        lines.append("")
     


    if "colmap" in contents:
       
        models = sorted(contents["colmap"].keys())
       
        for model_id in models:
 #           if model_id not in ["files"]:
 #               continue
            source_file = Path("projects")/ project / "colmap" / model_id / "sparse" / "points3D.ply"
            if not source_file.exists():
                continue
            destination_file = Path("docs") / "data" / project / f"{model_id}-colmap-points3D.ply"
            destination_file.parent.mkdir(parents=True, exist_ok=True)

            # Copy file
            if source_file.is_file():
                shutil.copy2(source_file, destination_file)
            else:
                continue
 
            lines.append(f"""
<div class="viewer-block">
<h3>Sparse Point Cloud: Model {model_id}</h3>
<div id="ply_viewer_{project}_{model_id}" class="threejs-container"></div>
</div>
""")
            
        lines.append(f"""

<script type="module">
import {{ loadPLYViewer }} from "../js/ply_viewer.js";
""")
        for model_id in models:
#            if model_id not in ["files"]:
#                continue
            source_file = Path("projects")/ project / "colmap" / model_id / "sparse" / "points3D.ply"
            if not source_file.exists():
                continue
            lines.append(f"loadPLYViewer(\"ply_viewer_{project}_{model_id}\", \"../data/{project}/{model_id}-colmap-points3D.ply\");");

        lines.append(f"</script>")
            
    
    out_file = dest / f"{project}.qmd"
    out_file.write_text("\n".join(lines))
    print(f"✅ Wrote: {out_file}")
    

def write_qmds_from_tree(tree, destination_folder, data_folder ):
    """
    Given a tree (from build_folder_tree_with_files) and a destination folder,
    writes a QMD file for each top-level project.
    """
    dest = Path(destination_folder)
    dest.mkdir(parents=True, exist_ok=True)


    write_index_qmd( tree, destination_folder, data_folder )
    
    for project in tree:
#        if not project.startswith("DJI"):
#            continue
        write_project_qmd(project, tree, destination_folder, data_folder)