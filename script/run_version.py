# Run for single d4j version
import os
import sys
import subprocess
ROOTDIR = "/root/CURE"
bugid = sys.argv[1]
proj, bid = bugid.split("-")

sys.path.append(os.path.join(ROOTDIR, "data", "data"))
import prepare_testing_data as ptd
sys.path.append(os.path.join(ROOTDIR, "src", "tester"))
import generator as gen
sys.path.append(os.path.join(ROOTDIR, "src", "validation"))
import rerank as rr

def init() -> None:
  os.makedirs(f"{ROOTDIR}/buggy", exist_ok=True)
  os.system(f'defects4j checkout -p {proj} -v {bid}b -w buggy/{bugid}')

def run_command_with_file(cmd: list, input_file: str, output_file: str) -> None:
  with open(input_file, "r") as in_f, open(output_file, "w") as ou_f:
    p = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdin=in_f, stdout=ou_f)
    out, err = p.communicate()
    print(err)

def prepare(d4j_dir: str, line_no: int, buggy_file: str, start_line: int) -> None:
  # d4j_dir: d4j/Chart-1
  tmp_dir = os.path.join(d4j_dir, "tmp", str(line_no), "tmp")
  out_dir = os.path.join(d4j_dir, "tmp", str(line_no), "out")
  os.makedirs(tmp_dir, exist_ok=True)
  os.makedirs(out_dir, exist_ok=True)
  ptd.prepare_cure_input(
        buggy_file=buggy_file,
        start_line=start_line,
        end_line=start_line + 1,
        java_class_path=f"{ROOTDIR}/data/data/java_class.json",
        java_keyword_path=f"{ROOTDIR}/data/data/java_keyword.json",
        tmp_dir=tmp_dir,
        output_dir=out_dir
      )
  """
  Run subword-nmt to perform subword tokenization
  subword-nmt apply-bpe -c ../vocabulary/subword.txt < input.txt > input_bpe.txt
  subword-nmt apply-bpe -c ../vocabulary/subword.txt < identifier.tokens > identifier_bpe.tokens
  """
  cmd = ["subword-nmt", "apply-bpe", "-c", f"{ROOTDIR}/data/vocabulary/subword.txt"]
  input_file = os.path.join(out_dir, "input.txt")
  input_bpe_file = os.path.join(out_dir, "input_bpe.txt")
  run_command_with_file(cmd, input_file, input_bpe_file)
  input_file = os.path.join(out_dir, "identifier.tokens")
  identifier_bpe_file = os.path.join(out_dir, "identifier_bpe.tokens")
  run_command_with_file(cmd, input_file, identifier_bpe_file)
  # run clean_testing_bpe() after running the subword-nmt commands above
  ptd.clean_testing_bpe(input_bpe_file, identifier_bpe_file)

def generate(d4j_dir: str, line_no: int) -> None:
  beam_size = 100
  vocab_file = os.path.join(ROOTDIR, "data", "vocabulary", "vocabulary.txt")
  out_dir = os.path.join(d4j_dir, "tmp", str(line_no), "out")
  input_file = os.path.join(out_dir, "input_bpe.txt")
  identifier_file = os.path.join(out_dir, "identifier.txt")
  identifier_tokens_file = os.path.join(out_dir, "identifier_bpe.tokens")
  
  model_file = os.path.join(ROOTDIR, "data", "models", "gpt_conut_1.pt")
  output_file = os.path.join(out_dir, "gpt_conut_1.txt")
  gen.generate_gpt_conut(vocab_file, model_file, input_file, identifier_file, identifier_tokens_file, output_file, beam_size)

  model_file = os.path.join(ROOTDIR, "data", "models", "gpt_fconv_1.pt")
  output_file = os.path.join(out_dir, "gpt_fconv_1.txt")
  gen.generate_gpt_fconv(vocab_file, model_file, input_file, identifier_file, identifier_tokens_file, output_file, beam_size)

def rerank(d4j_dir: str, line_no: int) -> None:
  outdir = os.path.join(d4j_dir, "tmp", str(line_no), "out")
  output_path = os.path.join(outdir, "reranked_patches.json")
  hypo_path_list = [os.path.join(outdir, "gpt_conut_1.txt"),
                    os.path.join(outdir, "gpt_fconv_1.txt")]
  meta = list()
  rr.cure_rerank(meta, hypo_path_list, output_path)

def run() -> None:
  print(f"Running {bugid}...")
  locationdir = '%s/location/ochiai/%s/%s.txt' % (ROOTDIR, proj.lower(), bid)
  line_no = -1
  init()
  dirs = os.popen(f'defects4j export -p dir.src.classes -w {ROOTDIR}/buggy/{bugid}').readlines()[-1]
  with open(locationdir, "r") as f:
    for line in f.readlines():
      line = line.strip()
      if len(line) == 0 or line.startswith("#"):
        continue
      line_no += 1
      if line_no > 10:
        break
      classname, others = line.split("#")
      if '$' in classname:
        classname = classname[:classname.index('$')]
      filepath = f"{ROOTDIR}/buggy/{bugid}/{dirs}/{classname.replace('.', '/')}.java"
      tokens = others.split(",")
      start_line = int(tokens[0])
      fl_score = float(tokens[1])
      d4j_dir = os.path.join(ROOTDIR, "d4j", bugid)
      # prepare
      prepare(d4j_dir, line_no, filepath, start_line)
      generate(d4j_dir, line_no)
      rerank(d4j_dir, line_no)

run()
    