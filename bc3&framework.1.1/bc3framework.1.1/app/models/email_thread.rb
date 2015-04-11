class EmailThread < ActiveRecord::Base
  has_many :emails
  has_many :summaries
  
  def myemails
    @myemails ||= emails.find(:all, :order => 'Date')
  end
  
  def summary(myexp)
    tmp = nil
    myexp.summaries.map { |s| tmp ||= s if s.email_thread == self}
    tmp
  end
  
end
