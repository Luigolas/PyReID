__author__ = 'luigolas'

import numpy as np
from itertools import islice, takewhile, count
import sys
import time


def safe_ln(x, minval=0.0000000001):
    return np.log(x.clip(min=minval))


def status(percent, flush=True):
    sys.stdout.write("%3d%%\r" % percent)
    if flush:
        sys.stdout.flush()
    else:
        sys.stdout.write("\n")


def split_every(n, it):
    """
    http://stackoverflow.com/a/22919323
    :param n:
    :param it:
    :return:
    """
    return takewhile(bool, (list(islice(it, n)) for _ in count(0)))


# split_every = (lambda n, it:
# takewhile(bool, (list(islice(it, n)) for _ in count(0))))


def chunks(iterable, n):
    """assumes n is an integer>0
    """
    iterable = iter(iterable)
    while True:
        result = []
        for i in range(n):
            try:
                a = next(iterable)
            except StopIteration:
                break
            else:
                result.append(a)
        if result:
            yield result
        else:
            break


def time_execution(fun, repeats=1):
    times = []
    for i in range(repeats):
        start_time = time.time()
        fun()
        end_time = time.time() - start_time
        times.append(end_time)
        print("Time %d --- %s seconds ---" % (i, end_time))
    print("Min time: --- %s seconds ---" % min(times))


def send_mail_by_gmail(user, password, files=None, subject="FluentMail", body=""):
    from fluentmail import FluentMail, TLS
    print "Sending mail: %s" % subject

    mail = FluentMail('smtp.gmail.com', 587, TLS)

    mail = mail.credentials(user, password) \
        .from_address(user) \
        .to(user) \
        .subject(subject) \
        .body(body, 'utf-8')
    if files:
        for f in files:
            mail.attach(f)
    mail.send()
    # .body(u'<h2>Hi, I\'m FluentMail.<h2>', 'utf-8') \


class Timer(object):
    """
    http://stackoverflow.com/a/5849861/3337586
    """

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print '[%s]' % self.name,
        print 'Elapsed: %s' % (time.time() - self.tstart)


# Exceptions definitions
# =====================
class InitializationError(Exception):
    pass


class ImagesNotFoundError(Exception):
    pass


class NotADirectoryError(Exception):
    pass


class FileNotFoundError(Exception):
    pass