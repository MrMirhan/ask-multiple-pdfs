# HTML templates for chat messages from the user and the bot and status of used Google search

css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem;
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 90px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  color: #fff;
}

.chat-message .google-search {
    color: #fff;
}

'''

bot_template = '''
<div class="chat-message bot container">
    <div class="row">
        <div class="avatar">
            <img src="https://www.isaca.org/-/media/images/isacadp/project/isaca/articles/press-releases/ai-pulse-poll_pr_550x550.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
        </div>
        <div class="col message">{{MSG}}</div>
    </div>
    <div class="row">
        <div class="google-search">{{GOOGLE_SEARCH}}</div>
    </div>
</div>
'''

user_template = '''
<div class="chat-message user container">
    <div class="row">
        <div class="avatar">
            <img src="https://upload.wikimedia.org/wikipedia/commons/9/99/Sample_User_Icon.png">
        </div>    
        <div class="col message">{{MSG}}</div>
    </div>
</div>
'''
