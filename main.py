from task_runner import TaskRunner
import oak_plotter

if __name__ == '__main__':
    tag = "base"
    tasks = {
        "algorithms" : ["random","linspacer"],
        "datasets" : ["indian_pines"],
        "target_sizes" : [5, 10, 15, 20, 25, 30]
    }
    ev = TaskRunner(tasks,10,tag,skip_all_bands=False)
    summary, details = ev.evaluate()
