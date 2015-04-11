class Experiment < ActiveRecord::Base
  belongs_to :participant
  has_many :summaries
  
  #can't change experiment after a certain time
  def still_valid?
    if Time.now - self.Date < 7 * 24 * 3600 #7 days
      return true
    else
      return false
    end
  end
end
