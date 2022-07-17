sample_rate = 100
sample_ply_file = open("sample"+str(sample_rate)+".ply", "w")

start_flag = False
i = 0
point_num = 0
points = []
model_idx = 0
seg_num = 0

for line in open("longdress2.ply", "r"):
    line = line.strip()
    if start_flag == False:
        if line == "end_header":
            print(line)
            start_flag = True
        line = line.split(" ")
        if line[0] == "element":
            print("Total Point Number: ", line[2])
            seg_num = int(line[2]) // 10 + 1
            print("Seg Length = ", seg_num)

    else:
        points.append(line+'\n')
        point_num += 1
        if point_num >= seg_num:
            filename = "segmodel/seg_model"+str(model_idx)+".ply"
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
                ''' % (len(points), "".join(points)))
            file.close()
            print("Length： ", len(points))

            point_num = 0
            points = []
            model_idx += 1

filename = "segmodel/seg_model"+str(model_idx)+".ply"
print("create sample file "+ filename)
file = open("segmodel/seg_model"+str(model_idx)+".ply", "w")
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
''' % (len(points), "".join(points)))
file.close()
print("Length： ", len(points))