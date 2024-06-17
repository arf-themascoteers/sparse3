from task_runner import TaskRunner
import oak_plotter

if __name__ == '__main__':
    tag = "nosig6"
    tasks = {
        "algorithms" : ["zhang_mean_fc_nosig6"],
        "datasets" : ["indian_pines"],
        "target_sizes" : list(range(5,31))
    }
    ev = TaskRunner(tasks,1,tag,skip_all_bands=False)
    summary, details = ev.evaluate()
