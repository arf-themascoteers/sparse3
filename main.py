from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "shapped"
    tasks = {
        "algorithms" : ["shapped"],
        "datasets" : ["indian_pines"],
        "target_sizes" : list(range(30,4,-1))
    }
    ev = TaskRunner(tasks,1,tag,skip_all_bands=False)
    summary, details = ev.evaluate()
