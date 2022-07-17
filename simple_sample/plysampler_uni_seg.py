sample_rate = 100
sample_ply_file = open("sample"+str(sample_rate)+".ply", "w")

start_flag = False
i = 0
point_num = 0
points_l = {}
model_idx = 0
seg_num = 0
slice_num = 64

for line in open("longdress2.ply", "r"):
    line = line.strip()
    if start_flag == False:
        if line == "end_header":
            print(line)
            for i in range(slice_num):
                points_l[i] = []
            start_flag = True
        line = line.split(" ")
        if line[0] == "element":
            print("Total Point Number: ", line[2])
            seg_num = int(line[2]) // slice_num + 1
            print("Seg Length = ", seg_num)

    else:
        points_l[point_num % slice_num].append(line+'\n')
        point_num += 1

for i in range(slice_num):
    filename = "uni_seg_model/seg_model"+str(i)+".ply"
    print("create sample file "+ filename)
    file = open(filename, "w")
    file.write('''ply
    format ascii 1.0
    element vertex %d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    %s
    ''' % (len(points_l[i]), "".join(points_l[i])))
    file.close()
    print("Length： ", len(points_l[i]))