import re
import os


def process_atomic_descriptions(takes, atomic_descriptions):
    def process_text(text):
        # Replace 'C' at the start of the string or followed by a space or punctuation, considering it as the camera wearer
        text = re.sub(r'\bC\b', 'the camera wearer', text)
        
        # Replace '0' surrounded by word boundaries, considering it as someone else
        text = re.sub(r'\b0\b', 'someone', text)
        
        return text
    atomic_descriptions_processed=[]
    unique_data=set()
    unprocessed = []
    for t_uid, annots in atomic_descriptions['annotations'].items():
        if t_uid not in takes:
            unprocessed.append(t_uid)
            continue
        take = takes[t_uid]
        root_dir = take['root_dir']
        annots = [a for a in annots if not a['rejected']]
        for ann in annots:
            ann['descriptions'] = [a for a in ann['descriptions'] if not a['unsure']]
            for disc in ann['descriptions']:
                disc['text'] = process_text(disc['text'])
                if disc['best_exo'] and disc['best_exo']['cam_id']:
                    disc['video_path'] = os.path.join(root_dir, take['frame_aligned_videos'][disc['best_exo']['cam_id']]['0']['relative_path'])
                    
                    if disc['text']+disc['video_path'] not in unique_data:
                        unique_data.add(disc['text']+disc['video_path'])
                        atomic_descriptions_processed.append({'take_uid': t_uid,
                                                              'video_path': disc['video_path'].replace('frame_aligned_videos', 'frame_aligned_videos/downscaled/448'),
                                                              'video_time': disc['timestamp'],
                                                              'duration':take['duration_sec'],
                                                              'id': t_uid,
                                                              'type':'atomic_description',
                                                              'text': disc['text'], 
                                                              'task_name':take['task_name']
                                                             })
                        
                if disc['ego_visible']:
                    ego_takes = {k:v for k,v in take['frame_aligned_videos'].items() if "aria" in k or "Aria" in k}
                    tmp=next(iter(ego_takes.values()))['rgb']
                    disc['video_path'] = os.path.join(root_dir, tmp['relative_path'])
                    if disc['text']+disc['video_path'] not in unique_data:
                        unique_data.add(disc['text']+disc['video_path'])
                        atomic_descriptions_processed.append({'take_uid': t_uid,
                                                              'video_path': disc['video_path'].replace('frame_aligned_videos', 'frame_aligned_videos/downscaled/448'),
                                                              'video_time': disc['timestamp'],
                                                              'duration':take['duration_sec'],
                                                              'id': t_uid,
                                                              'type':'atomic_description',
                                                              'text': disc['text'],
                                                              'task_name':take['task_name']
                                                             })
    return atomic_descriptions_processed, unprocessed


def process_expert_commentary(takes, expert_commentary):
    processed=[]
    unique_data=set()
    unprocessed = []
    for t_uid, annots in expert_commentary['annotations'].items():
        if t_uid not in takes:
            unprocessed.append(t_uid)
            continue
        take = takes[t_uid]
        root_dir = take['root_dir']
        for ann in annots:
            ann['commentary_data'] = [a for a in ann['commentary_data'] if not a['error']]
            for disc in ann['commentary_data']:
                if take['best_exo'] is not None:
                    disc['video_path'] = os.path.join(root_dir, take['frame_aligned_videos'][take['best_exo']]['0']['relative_path'])
                    if disc['text']+disc['video_path'] not in unique_data:
                        unique_data.add(disc['text']+disc['video_path'])
                        processed.append({'video_path': disc['video_path'].replace('frame_aligned_videos', 'frame_aligned_videos/downscaled/448'),
                                          'video_time': disc['video_time'],
                                          'text': disc['text'],
                                          'id': t_uid,
                                          'type':'expert_commentary',
                                          'duration':take['duration_sec'],
                                          'task_name':ann['task_name']
                                          })
                        
                ego_takes = {k:v for k,v in take['frame_aligned_videos'].items() if "aria" in k or "Aria" in k}
                tmp=next(iter(ego_takes.values()))['rgb']
                disc['video_path'] = os.path.join(root_dir, tmp['relative_path'])
                if disc['text']+disc['video_path'] not in unique_data:
                    unique_data.add(disc['text']+disc['video_path'])
                    processed.append({'video_path': disc['video_path'].replace('frame_aligned_videos', 'frame_aligned_videos/downscaled/448'),
                                      'video_time': disc['video_time'],
                                      'text': disc['text'],
                                      'id': t_uid,
                                      'type':'expert_commentary',
                                      'duration':take['duration_sec'],
                                      'task_name':ann['task_name']
                                      })
                            
    return processed, unprocessed


def process_proficiency_demonstration(takes, proficiency_demonstration):
    good_execution_data=[]
    tips_for_improvement_data=[]
    
    for annots in proficiency_demonstration['annotations']:     
        ego_vid = annots['video_paths']['ego']
        if annots['take_uid'] not in takes:
            exo_vid = [v for k, v in annots['video_paths'].items() if k != 'ego'][0]
        else:
            take = takes[annots['take_uid']]
            try:
                exo_vid = [vp for vp in annots['video_paths'].values() if take['best_exo'] in vp][0]
            except:
                exo_vid = [v for k, v in annots['video_paths'].items() if k != 'ego'][0]
        
        for good_execution in annots['good_executions']:
            good_execution_data.append({'video_time': good_execution['video_time'],
                                        'text': good_execution['list'],
                                        'task_name':annots['task_name'],
                                        'id': annots['take_uid'],
                                        'type':'good_execution',
                                        'duration':take['duration_sec'],
                                        'video_path': ego_vid.replace('frame_aligned_videos', 'frame_aligned_videos/downscaled/448')})
            
            good_execution_data.append({'video_time': good_execution['video_time'],
                                        'text': good_execution['list'],
                                        'duration':take['duration_sec'],
                                        'id': annots['take_uid'],
                                        'type':'good_execution',
                                        'task_name':annots['task_name'],
                                        'video_path': exo_vid.replace('frame_aligned_videos', 'frame_aligned_videos/downscaled/448')})

        for tips_for_improvement in annots['tips_for_improvement']:
            tips_for_improvement_data.append({'video_time': tips_for_improvement['video_time'],
                                              'text': good_execution['list'],
                                              'id': annots['take_uid'],
                                              'type':'tips_for_improvement',
                                              'duration':take['duration_sec'],
                                              'task_name':annots['task_name'],
                                              'video_path': ego_vid.replace('frame_aligned_videos', 'frame_aligned_videos/downscaled/448')})
            
            tips_for_improvement_data.append({'video_time': tips_for_improvement['video_time'],
                                              'text': good_execution['list'],
                                              'id': annots['take_uid'],
                                              'type':'tips_for_improvement',
                                              'duration':take['duration_sec'],
                                              'task_name':annots['task_name'],
                                              'video_path': exo_vid.replace('frame_aligned_videos', 'frame_aligned_videos/downscaled/448')})
                
    return good_execution_data, tips_for_improvement_data


def process_proficiency_demonstrator(takes, proficiency_demonstrator, atomic_description):
    good_execution_data=[]
    for annots in proficiency_demonstrator['annotations']:     
        ego_vid = annots['video_paths']['ego']
        if annots['take_uid'] not in takes:
            exo_vid = [v for k, v in annots['video_paths'].items() if k != 'ego'][0]
        else:
            take = takes[annots['take_uid']]
            try:
                exo_vid = [vp for vp in annots['video_paths'].values() if take['best_exo'] in vp][0]
            except:
                exo_vid = [v for k, v in annots['video_paths'].items() if k != 'ego'][0]

        descs = [a for a in atomic_description if a['take_uid']==annots['take_uid']]
        for disc in descs:
            good_execution_data.append({'task_name': annots['scenario_name'],
                                        'type': 'proficiency_demonstrator',
                                        'video_time': disc['video_time'],
                                        'id': annots['take_uid'],
                                        'text': disc['text'],
                                        'duration':take['duration_sec'],
                                        'proficiency': annots['proficiency_score'],
                                        'video_path': exo_vid.replace('frame_aligned_videos', 'frame_aligned_videos/downscaled/448')})

    return good_execution_data