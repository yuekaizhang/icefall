import logging
import argparse
from lhotse import CutSet, load_manifest_lazy

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--fixed-transcript-path",
        type=str,
        default="data/fbank/text.fix",
        help="""
        See https://github.com/wenet-e2e/WenetSpeech/discussions/54
        wget -nc https://huggingface.co/datasets/yuekai/wenetspeech_paraformer_fixed_transcript/resolve/main/text.fix
        """,
    )

    parser.add_argument(
        "--manifest-dir",
        type=str,
        default="data/fbank/",
        help="Directory to store the manifest files",
    )

    parser.add_argument(
        "--training-subset",
        type=str,
        default="L",
        help="The training subset for wenetspeech.",
    )

    return parser

def load_fixed_text(fixed_text_path):
    """
    fixed text format
    X0000016287_92761015_S00001 我是徐涛
    X0000016287_92761015_S00002 狄更斯的PICK WEEK PAPERS斯
    load into a dict
    """
    fixed_text_dict = {}
    with open(fixed_text_path, 'r') as f:
        for line in f:
            cut_id, text = line.strip().split(' ', 1)
            fixed_text_dict[cut_id] = text
    return fixed_text_dict

def fix_manifest(manifest, fixed_text_dict, fixed_manifest_path):
    with CutSet.open_writer(fixed_manifest_path) as manifest_writer:
        fixed_item = 0
        for i, cut in enumerate(manifest):
            if i % 10000 == 0:
                logging.info(f'Processing cut {i}, fixed {fixed_item}')
            cut_id_orgin = cut.id
            if cut_id_orgin.endswith('_sp0.9'):
                cut_id = cut_id_orgin[:-6]
            elif cut_id_orgin.endswith('_sp1.1'):
                cut_id = cut_id_orgin[:-6]
            else:
                cut_id = cut_id_orgin
            if cut_id in fixed_text_dict:
                if len(cut.supervisions) > 1:
                    print(cut)
                    print(233333333333333)
                    exit()
                assert len(cut.supervisions) == 1, f'cut {cut_id} has {len(cut.supervisions)} supervisions'
                if cut.supervisions[0].text != fixed_text_dict[cut_id]:
                    logging.info(f'Fixed text for cut {cut_id_orgin} from {cut.supervisions[0].text} to {fixed_text_dict[cut_id]}')
                    cut.supervisions[0].text = fixed_text_dict[cut_id]
                fixed_item += 1
                manifest_writer.write(cut)     

def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    parser = get_parser()
    args = parser.parse_args()
    logging.info(vars(args))

    operating_manifest_dir = args.manifest_dir
    manifest_path = operating_manifest_dir + f'cuts_{args.training_subset}.jsonl.gz'
    dev_manifest_path = operating_manifest_dir + 'cuts_DEV.jsonl.gz'
    fixed_text_path = operating_manifest_dir + 'text.fix'
    fixed_manifest_path = operating_manifest_dir + f'cuts_{args.training_subset}_fixed.jsonl.gz'
    fixed_dev_manifest_path = operating_manifest_dir + 'cuts_DEV_fixed.jsonl.gz'

    logging.info(f'Loading manifest from {manifest_path}')
    cuts_manifest = load_manifest_lazy(manifest_path)
    logging.info(f'Loading dev manifest from {dev_manifest_path}')
    cuts_dev_manifest = load_manifest_lazy(dev_manifest_path)

    fixed_text_dict = load_fixed_text(fixed_text_path)
    logging.info(f'Loaded {len(fixed_text_dict)} fixed texts')

    fix_manifest(cuts_dev_manifest, fixed_text_dict, fixed_dev_manifest_path)
    logging.info(f'Fixed dev manifest saved to {fixed_dev_manifest_path}')

    # tmp
    verify_dev_manifest_path = operating_manifest_dir + 'cuts_DEV_fixed_verify.jsonl.gz'
    cuts_dev_manifest = load_manifest_lazy(fixed_dev_manifest_path)
    fixed_text_dict = load_fixed_text(fixed_text_path)
    fix_manifest(cuts_dev_manifest, fixed_text_dict, verify_dev_manifest_path)
    logging.info(f'Fixed dev manifest saved to {verify_dev_manifest_path}')

    fix_manifest(cuts_manifest, fixed_text_dict, fixed_manifest_path)
    logging.info(f'Fixed manifest saved to {fixed_manifest_path}')



if __name__ == "__main__":
    main()