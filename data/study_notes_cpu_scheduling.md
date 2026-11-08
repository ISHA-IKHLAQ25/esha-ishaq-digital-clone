# My Study Notes: CPU Scheduling (the theory behind my simulator)

I wrote these notes while building my Process Scheduling Simulator. Writing them helped me understand the algorithms instead of memorising them, which is how I prefer to study.

## Why scheduling exists

A CPU core can run only one process at a time. When many processes are ready, the operating system needs a rule for choosing which one runs next. That rule is the scheduling algorithm. Different rules give different results for waiting time, turnaround time and fairness.

## Key terms

- **Arrival time:** when the process enters the ready queue.
- **Burst time:** how much CPU time the process needs.
- **Completion time:** when the process finishes.
- **Turnaround time** = completion time minus arrival time. The total time a process spends in the system.
- **Waiting time** = turnaround time minus burst time. The time a process spends waiting in the ready queue.
- **Preemptive vs non-preemptive:** in a preemptive algorithm the OS can take the CPU away from a running process; in a non-preemptive one, the process keeps the CPU until it finishes.

## FCFS: First Come, First Served

Non-preemptive. Processes run in order of arrival, like a queue at a bank. It is the simplest algorithm and easy to implement with a FIFO queue. The problem is the **convoy effect**: if a long process arrives first, all the short processes behind it have to wait, so the average waiting time can be high.

## SJF: Shortest Job First

Non-preemptive. Among the processes that have arrived, the one with the smallest burst time runs next. SJF gives the minimum average waiting time for a given set of processes, which is why it is called optimal. Its weaknesses: in real systems you do not know the burst time in advance, and long processes can **starve** if short ones keep arriving.

## SRTF: Shortest Remaining Time First

The preemptive version of SJF. Every time a new process arrives, the scheduler compares its burst time with the remaining time of the running process. If the new one is shorter, it takes the CPU. This reduces average waiting time even further but causes more context switches, and starvation is still possible.

## Priority scheduling

Each process has a priority number and the highest-priority process runs first. It can be preemptive or non-preemptive. The classic issue is starvation of low-priority processes. The usual solution is **aging**: slowly increasing the priority of processes that have been waiting a long time.

## Round Robin

Preemptive. Every process gets a fixed **time quantum**. If it does not finish within its quantum, it goes to the back of the ready queue and the next process runs. Round Robin is fair and gives good response time, which makes it suitable for time-sharing systems. The choice of quantum matters: if it is too large, Round Robin behaves like FCFS; if it is too small, the system wastes time on context switching.

## How I compare them in the simulator

The idea of my simulator is to run the same set of processes through each algorithm and compare the results side by side: the order of execution (like a Gantt chart), and the waiting and turnaround times. Seeing the numbers change from one algorithm to another makes the trade-offs obvious:

- FCFS is simple but can suffer from the convoy effect.
- SJF and SRTF minimise waiting time but can starve long jobs.
- Priority is flexible but needs aging to be fair.
- Round Robin is the fairest for interactive use but depends on a good quantum.

## What I took away

Building it myself taught me more than reading the slides. Once you have to write the code that picks the next process, you really understand what each algorithm is doing. That is my general rule: build it, and the concept sticks.
