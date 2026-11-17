"""Blender4.5 entry: exactly one authorized render, two output encodings."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.donut_camera0_reproduction import PROTOCOL,LIMITS,digest,require,verify_freeze,authorize


def capture_linear(tree,output):
    capture=tree.nodes.new('CompositorNodeOutputFile');capture.name='Camera0_Linear_Capture';capture.base_path=str(output/'HDR')
    capture.file_slots[0].path='scene-linear-';capture.format.file_format='OPEN_EXR';capture.format.color_depth='32';capture.format.color_mode='RGB';capture.format.exr_codec='ZIP'
    capture.format.color_management='OVERRIDE';capture.format.linear_colorspace_settings.name='Linear Rec.709';capture.save_as_render=False
    tree.links.new(tree.nodes['Reroute'].outputs[0],capture.inputs[0])
    require(capture.file_slots[0].use_node_format and capture.format.has_linear_colorspace,'Explicit linear EXR output required')
    return capture


def main():
    import bpy
    import numpy as np
    import OpenImageIO as oiio
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',type=Path,required=True);parser.add_argument('--freeze-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--authorization',type=Path);parser.add_argument('--commit')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    f=verify_freeze(args.freeze,args.freeze_sha256)
    require(bpy.app.version==(4,5,16) and bpy.app.build_hash.decode()=='3f6797307da5','Wrong reviewed Blender')
    require(not bpy.context.preferences.filepaths.use_scripts_auto_execute,'Scene autoexec must be disabled')
    bpy.ops.wm.open_mainfile(filepath=f['paths']['scene'],load_ui=False,use_scripts=False)
    scene=bpy.context.scene;camera=scene.camera
    meta=json.loads(Path(f['paths']['camera_JSON']).read_text())
    require(camera.name=='BlenderNeRF Camera' and np.array_equal(np.array(camera.matrix_world),np.array(meta['frames'][0]['transform_matrix'])) and
            camera.data.angle_x==meta['camera_angle_x'] and scene.frame_current==1,'Camera0/source-frame identity mismatch')
    require(scene.render.engine=='CYCLES' and scene.cycles.device=='CPU' and not scene.cycles.denoising_use_gpu and
            scene.cycles.samples==4096 and scene.cycles.use_adaptive_sampling and scene.cycles.use_denoising and
            (scene.render.resolution_x,scene.render.resolution_y,scene.render.resolution_percentage)==(1080,1080,100),
            'Original CPU render definition changed')
    require(scene.view_settings.view_transform=='AgX' and scene.view_settings.look=='None' and
            scene.view_settings.exposure==0 and scene.view_settings.gamma==1 and scene.display_settings.display_device=='sRGB','Wrong image formation')
    require(scene.render.image_settings.file_format=='PNG' and scene.render.image_settings.color_depth=='16' and
            scene.render.image_settings.color_mode=='RGBA' and not scene.render.film_transparent,'Original PNG/film changed')
    require(Path(bpy.path.abspath(bpy.data.images['university_workshop_4k.hdr'].filepath))==Path(f['paths']['HDR']),'HDR path changed; no relinking')
    tree=scene.node_tree
    require(scene.sequence_editor is None or not scene.sequence_editor.strips,'Unexpected sequencer image processing')
    composite=tree.nodes['Composite'];reroute=tree.nodes['Reroute']
    require(composite.inputs[0].links[0].from_node==reroute and
            reroute.inputs[0].links[0].from_node==tree.nodes['Render Layers'],'Unexpected compositor RGB path')
    depth=tree.nodes['File Output']
    require(len(depth.file_slots)==1 and depth.file_slots[0].path=='Image' and depth.file_slots[0].use_node_format and
            depth.format.file_format=='PNG' and depth.format.color_mode=='BW' and depth.format.color_depth=='16','Unexpected original depth output')
    if not args.execute:
        capture_linear(tree,args.output)
        im=oiio.ImageInput.open(f['paths']['reference_PNG']);pixels=im.read_image(format=oiio.UINT16);im.close()
        require(pixels.dtype==np.uint16 and pixels.shape==(1080,1080,4),'Lossless source PNG decoder required')
        print(json.dumps(dict(status='metadata_only',camera_matches=True,source_depth_slot=depth.file_slots[0].path,
            EXR_output_configured=True,source_PNG_dtype=str(pixels.dtype),scientific_render_calls=0)));return
    authorize(args.authorization,f,args.freeze_sha256,args.commit)
    output=args.output;require(str(output)==f['output_directory'] and output.is_dir() and not (output/'render-receipt.json').exists(),'Fresh owned render output required')
    def physics():
        cycles_keys=('device','samples','use_adaptive_sampling','adaptive_threshold','adaptive_min_samples','use_denoising','denoiser',
            'denoising_prefilter','denoising_quality','denoising_input_passes','denoising_use_gpu','max_bounces','diffuse_bounces','glossy_bounces',
            'transmission_bounces','volume_bounces','transparent_max_bounces','sample_clamp_direct','sample_clamp_indirect','seed','use_animated_seed',
            'pixel_filter_type','filter_width','blur_glossy','film_exposure','use_light_tree')
        shader=bpy.data.materials['Material.002'].node_tree.nodes['Principled BSDF']
        inputs={p.name:list(p.default_value) if hasattr(p.default_value,'__iter__') else p.default_value for p in shader.inputs if hasattr(p,'default_value')}
        light=bpy.data.objects['Light'];obj=bpy.data.objects['Torus.001']
        return dict(cycles={k:getattr(scene.cycles,k) for k in cycles_keys},material_inputs=inputs,distribution=shader.distribution,
            camera_world=[list(v) for v in camera.matrix_world],camera_angle=camera.data.angle_x,
            torus_world=[list(v) for v in obj.matrix_world],vertices=len(obj.data.vertices),polygons=len(obj.data.polygons),
            light=dict(energy=light.data.energy,color=list(light.data.color),radius=light.data.shadow_soft_size,world=[list(v) for v in light.matrix_world]),
            world_strength=scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value,
            view=dict(view=scene.view_settings.view_transform,look=scene.view_settings.look,exposure=scene.view_settings.exposure,gamma=scene.view_settings.gamma),
            render=dict(film_transparent=scene.render.film_transparent,dither=scene.render.dither_intensity,
                        PNG_color_depth=scene.render.image_settings.color_depth,PNG_color_mode=scene.render.image_settings.color_mode))
    before=physics();source_links={(l.from_node.name,l.from_socket.identifier,l.to_node.name,l.to_socket.identifier) for l in tree.links}
    for node in tree.nodes:
        if node.bl_idname=='CompositorNodeOutputFile':node.base_path=str(output/'source-depth')
    capture_linear(tree,output)
    scene.render.filepath=str(output/'reproduced.png')
    require(physics()==before,'Output routing changed physical settings')
    receipt=dict(schema_version=1,protocol=PROTOCOL,status='running',commit=args.commit,freeze_sha256=args.freeze_sha256,
        render_attempts=0,successful_render_returns=0,physical_settings_before=before,
        source_scene_sha256=digest(f['paths']['scene']),image_only_roundtrip=False,outputs={})
    def persist():
        (output/'render-receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    persist();started=time.monotonic()
    def read_image(path,format):
        im=oiio.ImageInput.open(str(path));require(im is not None,'Cannot open image: '+str(path))
        spec=im.spec();a=im.read_image(format=format);im.close();require(a is not None,'Image decode failed: '+str(path))
        return np.asarray(a),dict(width=spec.width,height=spec.height,channels=spec.nchannels,format=str(spec.format))
    try:
        receipt['render_attempts']=1;persist()
        returned=bpy.ops.render.render(write_still=True)  # The ONE permitted Cycles frame.
        receipt['operator_return']=sorted(returned)
        require(returned=={'FINISHED'},'Cycles operator did not finish')
        receipt['successful_render_returns']=1;receipt['render_wall_seconds']=time.monotonic()-started;persist()
        require(physics()==before and source_links<={(l.from_node.name,l.from_socket.identifier,l.to_node.name,l.to_socket.identifier) for l in tree.links},'Scientific state/compositor path changed')
        exr=output/'HDR/scene-linear-0001.exr';require(exr.is_file(),'Expected same-render EXR missing')
        linear,exr_spec=read_image(exr,oiio.FLOAT)
        require(linear.dtype==np.float32 and exr_spec['format']=='float' and linear.shape==(1080,1080,3) and np.isfinite(linear).all(),'32-bit finite linear RGB EXR required')
        image=bpy.data.images.load(str(exr),check_existing=False);image.colorspace_settings.name='Linear Rec.709'
        image.save_render(str(output/'linear-to-AgX.png'),scene=scene)  # Encoding-only; no second render.
        receipt['image_only_roundtrip']=True
        ref,ref_spec=read_image(Path(f['paths']['reference_PNG']),oiio.UINT16)
        pr,pr_spec=read_image(output/'reproduced.png',oiio.UINT16);rt,rt_spec=read_image(output/'linear-to-AgX.png',oiio.UINT16)
        require(ref_spec['format']==pr_spec['format']=='uint16' and ref.dtype==pr.dtype==rt.dtype==np.uint16,'Preserve16-bit PNG samples')
        mask,_=read_image(Path(f['paths']['reference_mask']),oiio.UINT8)
        dep,_=read_image(output/'source-depth/Image0001.png',oiio.UINT16)
        require(dep.shape[:2]==mask.shape[:2]==(1080,1080),'Depth/mask camera dimensions mismatch')
        np.savez_compressed(output/'comparison-inputs.npz',reference_RGB16=ref[...,:3],reproduced_RGB16=pr[...,:3],roundtrip_RGB16=rt[...,:3],
            linear_RGB=linear,reference_mask=mask.reshape(1080,1080)>127,render_mask=dep.reshape(1080,1080)<65535)
        verify_freeze(args.freeze,args.freeze_sha256)
        receipt.update(status='complete',physical_settings_after=physics(),source_unchanged=True,PNG_specs=dict(reference=ref_spec,reproduced=pr_spec,roundtrip=rt_spec),EXR_spec=exr_spec,
            outputs={p.relative_to(output).as_posix():dict(bytes=p.stat().st_size,sha256=digest(p)) for p in output.rglob('*') if p.is_file() and p.suffix in ('.png','.exr','.npz')})
    except BaseException as exc:
        receipt.update(status='failed',error_type=type(exc).__name__,error=str(exc));traceback.print_exc();raise
    finally:persist()


if __name__=='__main__':main()
