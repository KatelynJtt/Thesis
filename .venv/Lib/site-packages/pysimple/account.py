#!/usr/bin/python

# Copyright 2014 Tyler Fenby
#
# This file is part of PySimple.
#
# PySimple is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PySimple is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PySimple.  If not, see <http://www.gnu.org/licenses/>.

import requests
from lxml import html

from pysimple import exceptions

API = 'https://bank.simple.com'


class Account(object):
    """A Simple account

    Main entrypoint for the library
    """

    def __init__(self, username, password):
        self._username = username
        self._password = password
        self._session = requests.Session()
        self._csrf = None

    def login(self):
        """Log in to the bank via the API

        Must be done before any other commands will work

        :raises exceptions.LoginError: On any error in the request
        """
        getresp = self._session.get(API + '/signin')

        doc = html.fromstring(getresp.text)

        self._csrf = doc.xpath('//meta[@name="_csrf"]')[0].attrib['content']
        postresp = self._session.post(API + '/signin',
                                      data={
                                          'username': self._username,
                                          'password': self._password,
                                          '_csrf': self._csrf,
                                      })

        if postresp.url != 'https://bank.simple.com/activity':
            raise exceptions.LoginError

    @staticmethod
    def dollars_to_mills(dollars):
        """Convert dollars to mills, which Simple uses

        :param dollars: Amount to convert
        :type dollars: int, float
        :returns: Input amount, in mills
        :rtype: int
        """
        return int(dollars * 10000)

    @staticmethod
    def mills_to_dollars(mills):
        """Convert mills back to dollars

        :param int mills: Amount to convert
        :returns: Input amount, in mills
        :rtype: float
        """
        return mills / 10000.0

    def all_goals(self):
        """Returns all (active and archived) goals, raw from the API
        """
        resp = self._session.get(API + '/goals/data')
        return resp.json()

    def goals(self):
        """Return a list of active goals in a dict w/ their names as the key"""

        all_goals = self.all_goals()
        active = [x for x in all_goals if x['archived'] is False]
        by_name = {x['name']: x for x in active}
        return by_name

    def transactions(self):
        resp = self._session.get(API + '/transactions/data')
        return resp.json()

    def balances(self):
        resp = self._session.get(API + '/account/balances')
        return resp.json()

    def transfer_between_goals(self, from_id, to_id, amount):
        """Transfer between two goals

        :param from_id: ID of the source goal
        :param to_id: ID of the dest goal
        :param amount: Amount, in dollars, to transfer
        :raises requests.exceptions.RequestException: On any error
        """
        resp = self._session.post(API + '/goals/transfers',
                                  data={
                                      'from_goal_id': from_id,
                                      'to_goal_id': to_id,
                                      'amount': self.dollars_to_mills(amount),
                                      '_csrf': self._csrf,
                                  })
        resp.raise_for_status()

    def transfer_between_sts(self, to_id, amount):
        """Transfer between Safe-to-Spend and a goal

        A positive `amount` will transfer from StS to the goal,
        and a negative amount will do the opposite.

        :param amount: Amount, in dollars, to transfer
        :type amount: int, float
        :raises requests.exceptions.RequestException: On any error
        """
        resp = self._session.post(API + '/goals/%s/transactions' % to_id,
                                  data={
                                      'amount': self.dollars_to_mills(amount),
                                      '_csrf': self._csrf,
                                  })
        resp.raise_for_status()
