#!/bin/bash

# 设置显存阈值，单位为MB
threshold=4000

# 循环检查所有GPU的显存
while true
do
    # 获取所有GPU的显存使用情况
    gpu_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)

    # 将显存使用情况按空格分隔成数组
    gpu_memory_arr=($gpu_memory)

    # 设置计数器和最大显存变量
    counter=0
    max_memory=0

    # 循环遍历每张GPU的显存使用情况
    for i in "${!gpu_memory_arr[@]}"
    do
        # 获取显存使用情况
        memory=${gpu_memory_arr[$i]}

        # 判断显存是否大于阈值
        if [ $memory -gt $threshold ]; then
            # 如果显存大于阈值，则将计数器加1
            ((counter++))
            echo "GPU $i memory free: $memory MB"

            # 判断当前GPU的显存是否比最大显存更大
            if [ $memory -gt $max_memory ]; then
                # 如果当前GPU的显存更大，则更新最大显存变量和GPU编号变量
                max_memory=$memory
                gpu_number=$((i))
            fi
        fi
    done

    # 如果有一张以上的GPU显存大于阈值，则输出这些GPU中显存最大的GPU编号和显存使用情况
    if [ $counter -gt 0 ]; then
        echo "$counter GPU(s) have memory free greater than $threshold MB"
        break
    fi

    # 等待5秒后重新开始检查显存使用情况
    sleep 5
done

# 声明一个关联数组
declare -A map
# 给数组赋值
# map=([0]=2 [1]=3 [2]=0 [3]=1)
# 获取 gpu_number 的值
real_gpu_number=$gpu_number

echo "GPU: $real_gpu_number ($gpu_number) has the highest memory free: $max_memory MB"
# # 将变量输出到文件中
# echo $max_memory > max_memory.txt
# echo $gpu_number > gpu_number.txt

# 获取当前日期和时间，格式为 YYYY-MM-DD HH:MM:SS
current_time=$(date "+%Y-%m-%d %H:%M:%S")
# 将当前日期和时间追加到 time.txt 文件中
echo $current_time > bash_start_time.txt


tmux send-keys -t cby '' C-m;
tmux send-keys -t cby "export CUDA_VISIBLE_DEVICES=$real_gpu_number" C-m;
tmux send-keys -t cby '' C-m;
tmux send-keys -t cby 'cd /data/4TSSD/cby' C-m;
tmux send-keys -t cby '' C-m;
tmux send-keys -t cby 'conda activate plz_BY' C-m;
tmux send-keys -t cby '' C-m;
tmux send-keys -t cby 'cd /data/4TSSD/cby/Trans_G2/application' C-m;
tmux send-keys -t cby '' C-m;
tmux send-keys -t cby 'python demo.py' C-m;

