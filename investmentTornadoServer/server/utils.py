#!/usr/bin/env python
# -*- coding: utf-8 -*-
import socket


def l2m_str(l):
    return '{%s}' % ','.join(['"%s":' % a + str(b) for a, b in l])


def get_host_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip


if __name__ == '__main__':
    l = [('fa4d2d6d-b199-11e6-8836-005056b3f30e', 0.93), ('227948d3-b19a-11e6-8836-005056b3f30e', 0.123787622)]
    print l2m_str(l)
