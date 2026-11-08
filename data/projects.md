# My Projects

These are the main things I have built or am working on. I learn best by building, so my projects say more about my skills than my transcript does.

## Project 1: LoopTech360 Support (Customer Support / Helpdesk Web App)

**Type:** customer support / helpdesk web application.

**Tech stack:** React.js, Node.js, Supabase, Tailwind CSS, Antigravity (AI-assisted coding) and GitHub.

**What it is.** LoopTech360 Support is my recreation of the LoopTech360 Support website. I rebuilt its user interface, layout and typography, and I also built the full-stack functionality behind it, so it is not just a static copy of the look. A helpdesk app is the kind of system where users come with a problem and the support side has to receive it, track it and respond, so the front end and the back end both matter.

**How I approached it.** I started by studying the original site closely: how the pages are laid out, which fonts and sizes are used, how spacing works, and how the pages behave on different screen sizes. Then I recreated the interface in React.js with Tailwind CSS. For the back end I used Node.js, and Supabase handled the database side. I kept the code on GitHub. I used Antigravity, an AI-assisted coding tool, to speed up parts of the work, but I still had to understand and debug what was produced.

**What I learned.**

- **Reverse-engineering.** Recreating an existing website teaches you to look at a finished product and work backwards to how it was built. You notice small details you would normally ignore.
- **Responsive design.** Making the layout work on mobile, tablet and desktop taught me a lot about Tailwind's responsive utilities.
- **API and database integration.** Connecting the React front end to the back end and to Supabase was the most valuable part. Data has to flow correctly from the database to the screen and back again.

This is the project I would point to first if someone asked what kind of developer I am: practical, full-stack, and focused on getting a working result.

## Project 2: Process Scheduling Simulator

**Type:** full-stack educational tool.

**Tech stack:** Python with FastAPI on the back end, and HTML, CSS and JavaScript on the front end.

**What it does.** The Process Scheduling Simulator lets you simulate CPU scheduling algorithms from operating systems and compare them. It supports five algorithms:

- **FCFS (First Come, First Served):** processes run in the order they arrive. Simple, but one long process can make everyone else wait.
- **SJF (Shortest Job First):** the process with the shortest burst time runs next. It lowers average waiting time, but long jobs can starve.
- **SRTF (Shortest Remaining Time First):** the preemptive version of SJF. If a new process arrives with less remaining time, it takes over the CPU.
- **Priority scheduling:** the process with the highest priority runs first.
- **Round Robin:** each process gets a fixed time slice (quantum) in turn, which is fair and good for time-sharing.

**Why I built it.** Scheduling algorithms are easy to memorise and easy to forget. When you can enter processes and see the order they run in, and compare waiting and turnaround times between algorithms, the concepts actually make sense. Python and FastAPI do the scheduling logic, and the HTML/CSS/JS front end shows the results.

## Project 3: Cricket Ball Tracking System

**Type:** computer vision system.

**Tech stack:** Python, YOLO (Ultralytics), ONNX, and a custom trained model file called **best.pt**.

**What it does.** The Cricket Ball Tracking System detects and tracks a cricket ball in video footage. A cricket ball is small, moves very fast and is often blurred, so it is a hard object to detect. I used a YOLO model from Ultralytics, with a custom model (best.pt) trained for the cricket ball, to find the ball in each frame of the video. I also used ONNX, which is a portable model format that makes it easier to run a trained model efficiently in different environments.

**What I learned.** This project took me outside web development into machine learning and computer vision. I learned how object detection works frame by frame, why a custom model is needed for a specific object like a cricket ball, and how a trained model is exported and used for inference.

## Concepts and ongoing work

### GoSafe

GoSafe is a concept for a mobile app built with **React Native**. The idea is a **safe route finder**: instead of only showing the fastest way to get somewhere, it helps people find a safer route. It is still a concept / ongoing idea.

### Munshi

Munshi is a concept for a **WhatsApp-native AI accounting and POS (point of sale) SaaS**. The idea is that small shopkeepers and business owners already live on WhatsApp, so instead of making them learn complicated accounting software, Munshi would let them manage sales and accounts through WhatsApp with the help of AI. The name comes from the traditional "munshi", the person who keeps the accounts. It is still a concept.

## What ties my projects together

Looking at all of them together, I like building things that solve a real problem and that a real person can use: a helpdesk, a learning tool for scheduling, ball tracking for cricket, a safety app, and a simple accounting tool for small businesses. I care more about delivering something that works than about showing off.
