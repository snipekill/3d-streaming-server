

sample_rate = 5

for idx in range(1051, 1061):
    sample_ply_file = open("sample"+str(idx)+".ply", "w")
    start_flag = False
    i = 0
    point_count = 0
    for line in open("../models/longdress_vox10_"+str(idx) +".ply", "r"):
        i += 1
        line = line.strip()
        if start_flag == False:
            sample_ply_file.write(line + '\n')
            if line == "end_header":
                print(line)
                start_flag = True
        else:
            if i % sample_rate == 0:
                sample_ply_file.write(line + '\n')
                point_count += 1

    print(idx,"point Count:", point_count)
    sample_ply_file.close()