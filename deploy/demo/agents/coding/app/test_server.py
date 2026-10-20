import unittest

import server


class GreeterTest(unittest.TestCase):
    def test_greeting_capitalises_the_name(self):
        self.assertEqual(server.greeting("ada"), "Hello, Ada!")

    def test_the_page_greets_the_world(self):
        self.assertIn("<h1>Hello, World!</h1>", server.page())

    def test_the_page_names_its_build(self):
        self.assertIn("build ", server.page())


if __name__ == "__main__":
    unittest.main()
