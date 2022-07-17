


sample_ply_file = open("sample1_resize.ply", "w")

start_flag = False
i = 0
point_count = 0
for line in open("sample1_resize.ply", "r"):
    i += 1
    line = line.strip()

    if start_flag == False:
        sample_ply_file.write(line + '\n')
        if line == "end_header":
            print(line)
            start_flag = True
    else:
        if i % 2 == 0:
            sample_ply_file.write(line + '\n')
            point_count += 1

print(point_count)
sample_ply_file.close()