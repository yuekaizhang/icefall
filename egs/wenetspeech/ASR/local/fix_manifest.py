from lhotse import load_manifest, CutSet

operating_manifest_dir = '/workspace/icefall_zh/egs/wenetspeech/ASR/data/fbank/'
manifest_path = operating_manifest_dir + 'cuts_L.jsonl.gz'
dev_manifest_path = operating_manifest_dir + 'cuts_DEV.jsonl.gz'
fixed_text_path = operating_manifest_dir + 'text.fix'
fixed_manifest_path = operating_manifest_dir + 'cuts_L_fixed.jsonl.gz'
fixed_dev_manifest_path = operating_manifest_dir + 'cuts_DEV_fixed.jsonl.gz'

def load_fixed_text(fixed_text_path):
    """
    See https://github.com/wenet-e2e/WenetSpeech/discussions/54
    wget -nc https://huggingface.co/datasets/yuekai/wenetspeech_paraformer_fixed_transcript/resolve/main/text.fix
    fixed text format
    X0000016287_92761015_S00001 我是徐涛
    X0000016287_92761015_S00002 声动活泼
    load into a dict
    """
    fixed_text_dict = {}
    with open(fixed_text_path, 'r') as f:
        for line in f:
            cut_id, text = line.strip().split(' ')
            fixed_text_dict[cut_id] = text
            # add speed perturbation
            cut_id_sp_short = cut_id + '_sp0.9'
            fixed_text_dict[cut_id_sp_short] = text
            cut_id_sp_long = cut_id + '_sp1.1'
            fixed_text_dict[cut_id_sp_long] = text
    return fixed_text_dict


def fix_manifest(manifest, fixed_text_dict, fixed_manifest_path):
    fixed_cutset_list = []
    for cut in manifest:
        cut_id = cut.recording_id
        if cut_id in fixed_text_dict:
            if cut.supervisions[0].text != fixed_text_dict[cut_id]:
                print(f'Fixed text for cut {cut_id} from {cut.supervisions[0].text} to {fixed_text_dict[cut_id]}')
                cut.supervisions[0].text = fixed_text_dict[cut_id]
            fixed_cutset_list.append(cut)
    fixed_cutset = CutSet.from_cuts(fixed_cutset)
    fixed_cutset.to_file(fixed_manifest_path)        

if __name__ == '__main__':
    print(f'Loading manifest from {manifest_path}')
    cuts_manifest = load_manifest_lazy(manifest_path)
    print(f'Loading dev manifest from {dev_manifest_path}')
    cuts_dev_manifest = load_manifest_lazy(dev_manifest_path)
    fixed_text_dict = load_fixed_text(fixed_text_path)
    print(f'Loaded {len(fixed_text_dict)} fixed texts')
    fix_manifest(cuts_dev_manifest, fixed_text_dict, fixed_dev_manifest_path)
    print(f'Fixed dev manifest saved to {fixed_dev_manifest_path}')
    fix_manifest(cuts_manifest, fixed_text_dict, fixed_manifest_path)
    print(f'Fixed manifest saved to {fixed_manifest_path}')

    paths = [fixed_manifest_path, fixed_dev_manifest_path]
    for path in paths:
        print(f"Starting display the statistics for {path}")
        cuts = load_manifest_lazy(path)
        cuts.describe()
