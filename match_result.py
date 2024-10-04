from collections import defaultdict
from utils import *

match_dict = read_match_dict(dest_dir, epoch, 2)
print(match_dict)
cnt = defaultdict(int)
for survey_code in match_dict:
    cnt[match_dict[survey_code]] += 1

print(cnt)