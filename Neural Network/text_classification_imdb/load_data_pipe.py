s = """



These line is just for testing of loading data with different imput forms.

For example, content with empty lines, not like others with sentence stick together which is kind of results after processing.

As a result, these lines I type are seperate with empty lines.



"""

def Remove_space(s):
    """
    Modifying multiple lines string into one string with a proper form
    """
    result = list(s.strip())
    # print(result,"\n")
    for i in range(len(result)):
        if result[i] == '\n':
            result[i] = " "
    result = "".join(result)
    result = result.replace(".", "").replace(",", "").replace("(", "").replace(")", "").replace(":", "").replace("\n", "").replace("\"", "").split(" ")
    result = list(filter(("").__ne__, result))
    return result


# print(s)
# s = input("Type some sentence:\n")
s = """

Its one hell of a complicated film. It will be very hard for an average viewer to gather all the information provided by this movie at the first watch. But the more you watch it, more hidden elements will come to light. And when you are able to put these hidden elements together. You will realize that this movie is just a "masterpiece" which takes the legacy of Christopher Nolan Forward

If I talk about acting, Then I have to say that Robert Pattinson has really proved himself as a very good actor in these recent years. And I am sure his acting skills will increase with time. His performance is charming and very smooth. Whenever he is on the camera, he steals the focus John David Washington is also fantastic in this movie. His performance is electrifying, I hope to see more from him in the future. Other characters such as Kenneth Branagh, Elizabeth, Himesh Patel, Dimple Kapadia, Clémence Poésy have also done quite well. And I dont think there is a need to talk about Michael Caine

Talking about Music, its awesome. I dont think you will miss Hans Zimmer's score. Ludwig has done a sufficient job. There is no lack of good score in the movie

Gotta love the editing and post production which has been put into this movie. I think its fair to say this Nolan film has focused more in its post production. The main problem in the movie is the sound mixing. Plot is already complex and some dialogues are very soft due to the high music score. It makes it harder to realize what is going on in the movie. Other Nolan movies had loud BGM too. But Audio and dialogues weren't a problem

My humble request to everyone is to please let the movie sink in your thoughts. Let your mind grasp all the elements of this movie. I am sure more people will find it better. Even those who think they got the plot. I can bet they are wrong.
"""

# print(s)
print("\n")
# print(Remove_space(s))
filename = "load.txt"
with open(filename, encoding="utf-8") as f:
    content = f.read()
    print(content)
    print(type(content))
    print(Remove_space(content))




