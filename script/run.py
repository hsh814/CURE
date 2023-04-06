# Run for single d4j version
import os
import sys
import subprocess
import json
import time
import shutil
import javalang
from typing import List, Dict, Tuple, Set
ROOTDIR = os.path.abspath(os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1] + "..")

sys.path.append(os.path.join(ROOTDIR, "data", "data"))
import prepare_testing_data as ptd
sys.path.append(os.path.join(ROOTDIR, "src", "tester"))
import generator as gen
sys.path.append(os.path.join(ROOTDIR, "src", "validation"))
import rerank as rr
import validate_defects4j as vd4j
sys.path.append(os.path.join(ROOTDIR, "src", "dataloader"))
import tokenization

def get_end_line(node: javalang.tree.Node, lineid: int) -> int:
    line = lineid
    # print(type(node))
    if node is None or isinstance(node, str) or isinstance(node, bool):
        return line
    if isinstance(node, list) or isinstance(node, set):
        for n in node:
            line = get_end_line(n, line)
        return line   
    if hasattr(node, 'position'):
        if node.position is not None:
            if node.position.line > line:
                line = node.position.line
    if hasattr(node, 'children') and node.children is not None:
        for n in node.children:
            line = get_end_line(n, line)
    return line

def get_method_range(filename: str, lineid: int) -> dict:
    method_range = dict()
    found_method = False
    with open(filename, "r") as f:
        target = f.read()
        tokens = javalang.tokenizer.tokenize(target)
        parser = javalang.parser.Parser(tokens)
        tree = parser.parse()
        for path, node in tree.filter(javalang.tree.MethodDeclaration):
            if node.position is None:
                continue
            start_line = node.position.line
            end_line = get_end_line(node, start_line)
            if (start_line <= lineid + 1) and (end_line >= lineid + 1):
                print("found it!")
                print(f"{node.name} - {start_line}, {end_line}")
                method_range = { "function": node.name, "begin": start_line, "end": end_line }
                found_method = True
                break
        if found_method:
            return method_range
        for path, node in tree.filter(javalang.tree.ConstructorDeclaration):
            if node.position is None:
                continue
            start_line = node.position.line
            end_line = get_end_line(node, start_line)
            if (start_line <= lineid + 1) and (end_line >= lineid + 1):
                print("found it!")
                print(f"{node.name} - {start_line}, {end_line}")
                method_range = { "function": node.name, "begin": start_line, "end": end_line }
                found_method = True
                break
        if found_method:
            return method_range
        return { "function": "0no_function_found", "begin": lineid, "end": lineid }

def syntax_check(filename):
  try:
    with open(filename, 'r', encoding='utf-8') as f:
      tmpcode = f.read()
      tokens = javalang.tokenizer.tokenize(tmpcode)
      parser = javalang.parser.Parser(tokens)
      parser.parse()
  except:
    return False
  return True

def insert_fix_defects4j(original_file, start_loc, end_loc, patch, target_file):
  os.makedirs(os.path.dirname(target_file), exist_ok=True)
  shutil.copyfile(original_file, target_file)

  with open(target_file, 'r') as file:
    data = file.readlines()

  patched = False
  with open(target_file, 'w') as file:
    for idx, line in enumerate(data):
      if start_loc - 1 <= idx < end_loc - 1:
        if not patched:
          file.write(patch)
          patched = True
      else:
        file.write(line)
  return target_file

def validate_defects4j(file_name: str, line_no: int, fl_score: float, original_file: str, start_line: int, reranked_result_path: str, output_path: str, patch_dir: str):
  cnt = 0
  reranked_result = json.load(open(reranked_result_path, 'r'))
  dump_result = {}
  for key in reranked_result:
    dump_result = {'file': file_name, 'line': start_line, 'id': line_no, 'fl_score': fl_score, 'cases': []}
    bug_start_time = time.time()
    for tokenized_patch in reranked_result[key]['patches']:
      # validate 5000 patches for each bug at most
      if len(dump_result['cases']) >= 5000:
          break

      score = tokenized_patch['score']
      tokenized_patch = tokenized_patch['patch']

      strings, numbers = vd4j.get_strings_numbers(original_file, start_line)
      strings = [item[0] for item in strings][:5]
      numbers = [item[0] for item in numbers][:5]
      # one tokenized patch may be reconstructed to multiple source-code patches
      reconstructed_patches = tokenization.token2statement(tokenized_patch.split(' '), numbers, strings)
      # validate most 5 source-code patches come from the same tokenized patch
      for patch in reconstructed_patches[:5]:
        cnt += 1
        patch = patch.strip()
        patched_file = os.path.join(patch_dir, str(cnt), os.path.basename(original_file))
        insert_fix_defects4j(original_file, start_line, start_line + 1, patch, patched_file)
        if not syntax_check(patched_file):
          continue
        loc = os.path.join("patch", str(line_no), str(cnt), os.path.basename(original_file))
        obj = { 'case': cnt, 'location': loc, 'code': patch, 'prob': score }
        dump_result["cases"].append(obj)
  # write the last time after validating all
  json.dump(dump_result, open(output_path, 'w'), indent=2)


def run_command_with_file(cmd: list, input_file: str, output_file: str) -> None:
  with open(input_file, "r") as in_f, open(output_file, "w") as ou_f:
    p = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdin=in_f, stdout=ou_f)
    out, err = p.communicate()
    print(err)

def prepare(d4j_dir: str, line_no: int, buggy_file: str, start_line: int, end_line: int) -> None:
  # d4j_dir: d4j/Chart-1
  tmp_dir = os.path.join(d4j_dir, "tmp", str(line_no), "tmp")
  out_dir = os.path.join(d4j_dir, "tmp", str(line_no), "out")
  os.makedirs(tmp_dir, exist_ok=True)
  os.makedirs(out_dir, exist_ok=True)
  ptd.prepare_cure_input(
        buggy_file=buggy_file,
        start_line=start_line,
        end_line=end_line,
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

def generate(d4j_dir: str, line_no: int, beam_size: int) -> None:
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

def rerank(d4j_dir: str, line_no: int, meta: list) -> None:
  outdir = os.path.join(d4j_dir, "tmp", str(line_no), "out")
  output_path = os.path.join(outdir, "reranked_patches.json")
  hypo_path_list = [os.path.join(outdir, "gpt_conut_1.txt"),
                    os.path.join(outdir, "gpt_fconv_1.txt")]
  rr.cure_rerank(meta, hypo_path_list, output_path)

def dump(d4j_dir: str, line_no: int, fl_score: float, file_name: str, original_file: str, start_line: int) -> None:
  patch_dir = os.path.join(d4j_dir, "patch", str(line_no))
  outdir = os.path.join(d4j_dir, "tmp", str(line_no), "out")
  ranked_file = os.path.join(outdir, "reranked_patches.json")
  output_file = os.path.join(outdir, "dumped_patches.json")
  validate_defects4j(file_name, line_no, fl_score, original_file, start_line, ranked_file, output_file, patch_dir)

# def get_func_map(locations: list) -> list:
#   func_map: Dict[str, List[dict]] = dict()
#   for file, line_number, fl_score in locations:
#     real_file = os.path.join(ROOTDIR, "buggy", bugid, file)
#     if file not in func_map:
#       func_map[file] = list()
#     try:
#       func_loc = get_method_range(real_file, line_number)
#       func_map[file].append(func_loc)
#     except Exception:
#       continue
#   func_locations = list()
#   for file in func_map:
#     func_filter = dict()
#     functions = list()
#     tmp_file_level = { "file": file, "functions": functions }
#     for func in func_map[file]:
#       func_id = f"{func['function']}:{func['begin']}-{func['end']}"
#       if func_id not in func_filter:
#         func_filter[func_id] = func
#         functions.append(func)
#     func_locations.append(tmp_file_level)
#   return func_locations

def add_tests(outdir: str, bugid: str, switch_info: dict) -> None:
  proj = bugid.split("-")[0]
  bid = bugid.split("-")[1]
  build_dir = os.path.join(ROOTDIR, "buggy", bugid)
  gen_fixed_proj_cmd = f"defects4j checkout -p {proj} -v {bid}f -w {build_dir}f"
  os.system(gen_fixed_proj_cmd)
  compile_fixed = f"defects4j compile -w {build_dir}f"
  os.system(compile_fixed)
  fix_test_cmd = ["defects4j", "test", "-w", build_dir + "f"]
  test_proc = subprocess.Popen(fix_test_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  so, se = test_proc.communicate()
  result_str = so.decode("utf-8")
  err_str = se.decode("utf-8")
  failed_tests = list()
  for line in result_str.splitlines():
    line = line.strip()
    if line.startswith("Failing tests:"):
      error_num = int(line.split(":")[1].strip())
      continue
    if line.startswith("-"):
      ft = line.replace("-", "").strip()
      failed_tests.append(ft)
  tests_all_file = os.path.join(outdir, "tests.all")
  tests_relevant_file = os.path.join(outdir, "tests.rel")
  tests_trigger_file = os.path.join(outdir, "tests.trig")
  gen_test_all_cmd = f"defects4j export -w {build_dir} -o {tests_all_file} -p tests.all"
  os.system(gen_test_all_cmd)
  gen_test_rel_cmd = f"defects4j export -w {build_dir} -o {tests_relevant_file} -p tests.relevant"
  os.system(gen_test_rel_cmd)
  gen_test_trig_cmd = f"defects4j export -w {build_dir} -o {tests_trigger_file} -p tests.trigger"
  os.system(gen_test_trig_cmd)
  # TODO: tests
  failing_test_cases = list()
  failed = dict()
  passing_test_cases = list()
  relevent_test_cases = list()
  with open(tests_trigger_file, "r") as tf:
    for line in tf.readlines():
      test = line.strip()
      failing_test_cases.append(test)
  with open(tests_all_file, "r") as tf:
    for line in tf.readlines():
      test = line.strip()
      passing_test_cases.append(test)
  with open(tests_relevant_file, "r") as tf:
    for line in tf.readlines():
      test = line.strip()
      relevent_test_cases.append(test)
  switch_info["failing_test_cases"] = failing_test_cases
  switch_info["passing_test_cases"] = passing_test_cases
  switch_info["relevant_test_cases"] = relevent_test_cases
  switch_info["failed_passing_tests"] = failed_tests


def collect_patches(bugid: str, d4j_dir: str) -> None:
  obj = { "project_name": bugid }
  add_tests(os.path.join(d4j_dir, "tmp"), bugid, obj)
  file_map = dict()
  patch_ranking = list()
  for i in range(max_line_no):
    dumped_patch_file = os.path.join(d4j_dir, "tmp", str(i), "out", "dumped_patches.json")
    if os.path.exists(dumped_patch_file):
      with open(dumped_patch_file, "r") as f:
        line_info = json.load(f)
        file_name = line_info["file"]
        line_num = line_info["line"]
        line_id = line_info["id"]
        if file_name not in file_map:
          file_map[file_name] = list()
        file_map[file_name].append(line_info)
        for patch in line_info["cases"]:
          patch_ranking.append(f"{line_id}-{patch['case']}")
  rules = list()
  for file_name in file_map:
    file_info = { "file": file_name, "lines": file_map[file_name] }
    rules.append(file_info)
  obj["rules"] = rules
  obj["ranking"] = patch_ranking
  # obj["func_locations"] = get_func_map(locations)
  with open(os.path.join(d4j_dir, "switch-info.json"), "w") as f:
    json.dump(obj, f, indent=2)

def get_meta(meta_file: str) -> dict:
  result = dict()
  with open(meta_file, "r") as f:
    for line in f.readlines():
      line = line.strip()
      if len(line) == 0 or line.startswith("#"):
        continue
      tokens = line.split()
      if len(tokens) < 5:
        continue
      proj = tokens[0]
      bid = tokens[1]
      file = tokens[2]
      start = int(tokens[3])
      end = int(tokens[4])
      obj = { "file": file, "start": start, "end": end }
      result[f"{proj}-{bid}"] = obj
  return result

def run(args) -> None:
  print(f"Running {args.bug_id}...")
  proj, bid = args.bug_id.split("-")
  bugid = f"{proj}-{bid}"
  os.makedirs(f"{ROOTDIR}/buggy", exist_ok=True)
  os.system(f'defects4j checkout -p {proj} -v {bid}b -w buggy/{bugid}')
  os.system(f"rm -r {ROOTDIR}/d4j/{bugid}")
  meta = os.path.join(ROOTDIR, "candidate_patches", "Defects4Jv1.2", "meta.txt")
  meta_map = get_meta(meta)
  info = meta_map[bugid]
  # dirs = os.popen(f'defects4j export -p dir.src.classes -w {ROOTDIR}/buggy/{bugid}').readlines()[-1]
  filename = info["file"]
  filepath = os.path.join(ROOTDIR, "buggy", bugid, filename)
  start_line = info["start"]
  end_line = info["end"]
  d4j_dir = os.path.join(ROOTDIR, "d4j", bugid)
  line_no = 0
  fl_score = 1.0
  meta = [[proj, bid, filename, str(start_line), str(end_line)]]
  prepare(d4j_dir, line_no, filepath, start_line)
  generate(d4j_dir, line_no, args.beam_size)
  rerank(d4j_dir, line_no, meta)
  dump(d4j_dir, line_no, fl_score, filename, filepath, start_line)    
  # collect result to switch_info file
  collect_patches(bugid, d4j_dir)
 

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--bug_id", type=str, required=True)
  parser.add_argument("--output_dir", type=str, required=True)
  parser.add_argument("--beam_size", type=int, default=50)
  parser.add_argument("--template_model", type=str, default="recoder")
  parser.add_argument("--nmt_model", type=str, default="codex")
  parser.add_argument("--template_engine", type=str, default="gumtree")
  parser.add_argument("--skip_template_gen", dest="skip_template_gen", action='store_true', default=False)
  parser.add_argument("--template_path", type=str, default=None)
  parser.add_argument("--benchmark", type=str, default="Defects4J")
  args = parser.parse_args()
  run(args)
