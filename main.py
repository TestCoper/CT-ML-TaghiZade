import os
NormalScans = [
    os.path.join(os.getcwd(), "/root/CT-0", x)
    for x in os.listdir('/root/CT-0/')
]

AbnormalScans = [
    os.path.join(os.getcwd(), "/root/CT-23", x)
    for x in os.listdir("/root/CT-23")
]

print("CT scans with normal lung tissue: " + str(len(NormalScans)))
print("CT scans with abnormal lung tissue: " + str(len(AbnormalScans)))