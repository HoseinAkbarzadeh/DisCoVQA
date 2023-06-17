from types import TracebackType
from typing import (
    Optional,
    Type
)

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress, 
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn
)

class RichProgressBar:
    def __init__(self, **kwargs) -> None:
        extra = []
        self.initials = kwargs
        for arg in self.initials.keys():
            extra.append("|")
            extra.append(TextColumn("{}: {{task.fields[{}]:0.4f}}".format(arg, arg), justify='left'))

        self.progress = Progress(
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            *extra
        )

        self.train_total = 0
        self.val_total = 0

    def __enter__(self):
        return self.progress.__enter__()

    def __exit__(self, exc_type: Optional[Type[BaseException]],
                        exc_val: Optional[BaseException],
                        exc_tb: Optional[TracebackType]) -> None:
        return self.progress.__exit__(exc_type, exc_val, exc_tb)
    
    def start_training(self, total_epochs):
        self.train_task = self.progress.add_task("Training progress bar", **self.initials)

    def start_validation(self, total_itr):
        self.val_task = self.progress.add_task("Validation progress bar", **self.initials)

    
    
    
if __name__ == '__main__':
    from time import sleep
    bar = RichProgressBar(train_loss=0.0, val_loss=0.0)

    train_loss = 0.0
    val_loss = 0.0
    with bar as p:
        task_id = p.add_task("New task", train_loss=train_loss, val_loss=val_loss)
        for i in range(100):
            p.update(task_id=task_id, completed=i, total=100, train_loss=train_loss, val_loss=val_loss)
            if i == 99:
                task_2 = p.add_task("Second task", val_loss=val_loss, train_loss=train_loss)
                for j in range(50):
                    p.update(task_2, completed=j, total=50, val_loss=val_loss, train_loss=train_loss)
                    val_loss += 0.1
                    sleep(0.1)
                p.remove_task(task_2)
            train_loss += 0.1
            sleep(0.1)
        p.reset(task_id)
        for i in range(100):
            p.update(task_id=task_id, completed=i, total=100, train_loss=train_loss, val_loss=val_loss)
            if i == 99:
                task_2 = p.add_task("Second task", val_loss=val_loss, train_loss=train_loss)
                for j in range(50):
                    p.update(task_2, completed=j, total=50, val_loss=val_loss, train_loss=train_loss)
                    val_loss += 0.1
                    sleep(0.1)
                p.remove_task(task_2)
            train_loss += 0.1
            sleep(0.1)